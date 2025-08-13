# distill_hidden_states.py
# Full, corrected training script for hidden-state KD:
# - Assistant-only XE labels (pads/user/system masked to -100)
# - Adaptation layer is TRAINED (added to optimizer)
# - KD computed with attention-mask intersection (no pad noise)
# - Teacher uses its native chat template (Harmony) and trust_remote_code
# - Context length set to min(student, teacher)
# - Clean HF Trainer usage (no double-accelerate)

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, interleave_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer

# -------------------------------
# Repro & env hygiene
# -------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_PROJECT"] = "distil-multilayer"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Enable TF32 for faster matmul on A100/H100
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# -------------------------------
# Config
# -------------------------------
config = {
    "dataset": {
        "name": "mlabonne/FineTome-100k",  # unused with mixture loading
        "split": "train",                 # unused with mixture loading
        "num_samples": 10,               # unused with mixture loading  
        "seed": SEED,
    },
    "models": {
        "teacher": "openai/gpt-oss-20b",
        "student": "HuggingFaceTB/SmolLM2-1.7B",
    },
    "tokenizer": {
        # will be replaced with min(student_max, teacher_initial_ctx) below
        "max_length": 4096,
        # Student chat template (SmolLM2 is Llama-like; using your original tags)
        "student_chat_template": (
            "{% for message in messages %}"
            "{% if loop.first and messages[0]['role'] != 'system' %}"
            "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
            "{% endif %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        ),
        # Tokens that mark assistant spans in the student format
        "assistant_start_text": "<|im_start|>assistant\n",
        "assistant_end_text": "<|im_end|>",
    },
    "training": {
        "output_dir": "./results",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 16,
        "save_steps": 1000,
        "logging_steps": 2,
        "save_total_limit": 2,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "linear",
        "resume_from_checkpoint": None,
        "bf16": True,
        "fp16": False,
        "max_grad_norm": 0.5,
        "group_by_length": True,  # enable dynamic padding efficiency
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "dataloader_pin_memory": False,
        "dataloader_num_workers": 2,
    },
    "distillation": {
        "alpha": 0.7,            # weight on KD vs XE
        "loss_type": "cosine",   # "cosine" | "mse" | "huber"
        "use_pooled_kd": False,  # fallback for tokenization mismatch
        "last_k_layers": 8,      # None = all layers, int = last K layers (8 recommended for stability)
    },
    "model_config": {
        "use_flash_attention": False,  # set True only if your env has FA2 installed
        "torch_compile": False,        # can switch to True on H100 once stable
        "use_cache": False,            # disable KV cache during training
    },
}

# -------------------------------
# Mixture adapters (map various schemas -> ShareGPT-style "conversations")
# -------------------------------
def _conv(human_text, assistant_text):
    return {"conversations": [
        {"from": "human", "value": human_text if human_text is not None else ""},
        {"from": "gpt",   "value": assistant_text if assistant_text is not None else ""},
    ]}

def map_sharegpt(example):
    # Already in conversations format
    return {"conversations": example["conversations"]}

def map_alpaca(example):
    # fields: instruction / input / output (input may be empty)
    instr = example.get("instruction", "")
    inp   = example.get("input", "")
    user  = instr if not inp else f"{instr}\n\nInput:\n{inp}"
    return _conv(user, example.get("output", ""))

def map_qa(example):
    # fields: question / answer
    return _conv(example.get("question", ""), example.get("answer", ""))

def map_code_instruct(example):
    # common patterns: prompt/response OR instruction/response OR question/solution
    user = (example.get("prompt") or
            example.get("instruction") or
            example.get("question") or "")
    asst = (example.get("response") or
            example.get("completion") or
            example.get("solution") or
            example.get("output") or "")
    return _conv(user, asst)

def map_messages(example):
    # fields: messages = [{"role":"user"/"assistant"/"system","content":...}, ...]
    conv = []
    for m in example.get("messages", []):
        if m.get("role") == "user":
            conv.append({"from": "human", "value": m.get("content", "")})
        elif m.get("role") == "assistant":
            conv.append({"from": "gpt", "value": m.get("content", "")})
    if not conv:
        return map_alpaca(example)
    return {"conversations": conv}

# -------------------------------
# Recommended production mixture (weights roughly sum to 1.0)
# Tweak weights to steer training signal (see notes below).
# -------------------------------
MIXTURE = [
    # General chat/knowledge (keeps your current behavior)
    {"name": "mlabonne/FineTome-100k", "split": "train", "weight": 0.45, "adapter": map_sharegpt},

    # General instruction-following (broad coverage; messages schema)
    {"name": "HuggingFaceH4/ultrachat_200k", "split": "train_sft", "weight": 0.20, "adapter": map_messages},

    # Math (CoT-style reasoning; boosts AIME/HiddenMath/ARC-c/GPQA)
    {"name": "meta-math/MetaMathQA", "split": "train", "weight": 0.10, "adapter": map_qa},
    {"name": "nvidia/OpenMathInstruct-2", "split": "train", "weight": 0.10, "adapter": map_alpaca},

    # Code instruction (boosts LCB/MBPP/HumanEval/CodeGolf)
    {"name": "ise-uiuc/Magicoder-OSS-Instruct-75K", "split": "train", "weight": 0.15, "adapter": map_code_instruct},
]

def load_mixture(mixture, seed=42, target_total=None):
    shards = []
    for spec in mixture:
        ds = load_dataset(spec["name"], split=spec.get("split", "train"))
        if target_total is not None:
            n = int(target_total * float(spec.get("weight", 1.0)))
            n = min(n, len(ds))
            ds = ds.shuffle(seed=seed).select(range(n))
        ds = ds.map(spec["adapter"], remove_columns=ds.column_names)
        
        # keep only examples that produced ≥2 turns
        def _ok(e):
            c = e.get("conversations", [])
            return isinstance(c, list) and len(c) >= 2
        ds = ds.filter(_ok)
        
        shards.append(ds)
    
    # sizes are already set to your weights; use all samples
    return interleave_datasets(shards, seed=seed, stopping_strategy="all_exhausted")

# -------------------------------
# Data (mixture)
# Choose target size based on your goals:
# - Smoke test: target_total=10_000 (~10 mins)
# - Baseline: target_total=300_000 (~200-300M tokens, solid gains)
# - Production: target_total=500_000 (~350M-1B tokens, better capability)
# - Max: target_total=1_000_000 (~1B+ tokens, diminishing returns)
# -------------------------------
dataset = load_mixture(MIXTURE, seed=config["dataset"]["seed"], target_total=300_000)

# -------------------------------
# Tokenizers
# -------------------------------
# Teacher uses Harmony format; requires trust_remote_code and (usually) kernels installed
teacher_tokenizer = AutoTokenizer.from_pretrained(
    config["models"]["teacher"],
    trust_remote_code=True,
)
student_tokenizer = AutoTokenizer.from_pretrained(
    config["models"]["student"],
)

# Ensure PAD tokens exist
if teacher_tokenizer.pad_token is None:
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
if student_tokenizer.pad_token is None:
    student_tokenizer.pad_token = student_tokenizer.eos_token

# Apply student chat template (teacher has its own Harmony template already on the repo)
student_tokenizer.chat_template = config["tokenizer"]["student_chat_template"]

# -------------------------------
# Models
# -------------------------------
# Choose dtype
torch_dtype = (
    torch.bfloat16
    if config["training"]["bf16"]
    else (torch.float16 if config["training"]["fp16"] else torch.float32)
)

student_model_kwargs = {
    "torch_dtype": torch_dtype,
    "use_cache": config["model_config"]["use_cache"],
}
if config["model_config"]["use_flash_attention"]:
    student_model_kwargs["attn_implementation"] = "flash_attention_2"

# Teacher has a custom architecture; trust_remote_code is recommended
teacher_model_kwargs = {
    "torch_dtype": torch_dtype,
    "use_cache": config["model_config"]["use_cache"],
    "trust_remote_code": True,
}

teacher_model = AutoModelForCausalLM.from_pretrained(
    config["models"]["teacher"],
    **teacher_model_kwargs,
)
student_model = AutoModelForCausalLM.from_pretrained(
    config["models"]["student"],
    **student_model_kwargs,
)

# Keep teacher in eval & no grad forever
teacher_model.eval()
for p in teacher_model.parameters():
    p.requires_grad = False

# Disable cache at config level (some models check config too)
student_model.config.use_cache = False
teacher_model.config.use_cache = False

# (Optional) compile on supported GPUs once stable
if config["model_config"]["torch_compile"]:
    student_model = torch.compile(student_model, mode="max-autotune")

# -------------------------------
# Context length sanity: min(student, teacher)
# Teacher config (original/config.json) exposes "initial_context_length"
# Student config shows "max_position_embeddings"
# -------------------------------
def get_teacher_initial_ctx_fallback(default_ctx=4096):
    # best-effort: try to read from huggingface config if present
    # gpt-oss exposes it in original/config.json; the loaded config may or may not carry it
    return getattr(teacher_model.config, "initial_context_length", default_ctx)

student_max_ctx = getattr(student_model.config, "max_position_embeddings", 4096)
teacher_init_ctx = get_teacher_initial_ctx_fallback(4096)
max_ctx = int(min(student_max_ctx, teacher_init_ctx, config["tokenizer"]["max_length"]))
config["tokenizer"]["max_length"] = max_ctx

# -------------------------------
# Prepare samples (no padding here; collator will handle dynamic padding)
# -------------------------------
def build_messages(example):
    system = "You are a helpful assistant chatbot."
    conversations = example["conversations"]
    msgs = [{"role": "system", "content": system}]
    for turn in conversations:
        if turn.get("from") == "human":
            msgs.append({"role": "user", "content": turn.get("value", "")})
        elif turn.get("from") == "gpt":
            msgs.append({"role": "assistant", "content": turn.get("value", "")})
    return msgs

def prepare_dataset(example):
    messages = build_messages(example)

    # Student text (your <|im_start|>… format)
    student_text = student_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Teacher text (Harmony format; teacher tokenizer has its jinja in the repo)
    teacher_text = teacher_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    student_enc = student_tokenizer(
        student_text, truncation=True, max_length=max_ctx, padding=False
    )
    teacher_enc = teacher_tokenizer(
        teacher_text, truncation=True, max_length=max_ctx, padding=False
    )

    return {
        "input_ids": student_enc["input_ids"],
        "attention_mask": student_enc["attention_mask"],
        # labels will be built in the collator so masking aligns with padded sequences
        "teacher_input_ids": teacher_enc["input_ids"],
        "teacher_attention_mask": teacher_enc["attention_mask"],
    }

print("Preprocessing and tokenizing dataset...")
original_columns = dataset.column_names
dataset = dataset.map(prepare_dataset, remove_columns=original_columns)

# -------------------------------
# Collator with assistant-only labels (+ dynamic padding)
# -------------------------------
@dataclass
class KDDataCollator:
    tokenizer: AutoTokenizer
    teacher_tokenizer: AutoTokenizer
    assistant_start_text: str
    assistant_end_text: str
    max_length: Optional[int] = None  # None = dynamic padding to batch max

    def _find_span_labels(self, input_ids: List[int]) -> List[int]:
        # Build -100 labels except within assistant spans defined by the student text markers
        start_ids = self.tokenizer(self.assistant_start_text, add_special_tokens=False)["input_ids"]
        end_ids = self.tokenizer(self.assistant_end_text, add_special_tokens=False)["input_ids"]
        L = len(input_ids)
        labels = [-100] * L
        i = 0
        while i <= L - len(start_ids):
            if input_ids[i : i + len(start_ids)] == start_ids:
                j = i + len(start_ids)
                # label until the next <|im_end|> (or end if truncated)
                while j < L and not (j <= L - len(end_ids) and input_ids[j:j+len(end_ids)] == end_ids):
                    labels[j] = input_ids[j]
                    j += 1
                i = j
            i += 1
        return labels

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Gather
        batch = {
            "input_ids": [f["input_ids"] for f in features],
            "attention_mask": [f["attention_mask"] for f in features],
            "teacher_input_ids": [f["teacher_input_ids"] for f in features],
            "teacher_attention_mask": [f["teacher_attention_mask"] for f in features],
        }
        # Build labels BEFORE padding, then pad and align
        labels_list = [self._find_span_labels(ids) for ids in batch["input_ids"]]

        # Decide target length for this batch (dynamic)
        if self.max_length is None:
            max_len = max(
                max(len(x) for x in batch["input_ids"]),
                max(len(x) for x in batch["teacher_input_ids"]),
            )
            max_len = min(max_len, config["tokenizer"]["max_length"])
        else:
            max_len = self.max_length

        def pad_to(x: List[int], pad_id: int, to_len: int) -> List[int]:
            if len(x) < to_len:
                return x + [pad_id] * (to_len - len(x))
            return x[:to_len]

        padded = {}
        # Student tensors
        pad_id = self.tokenizer.pad_token_id or 0
        padded["input_ids"] = torch.tensor([pad_to(x, pad_id, max_len) for x in batch["input_ids"]], dtype=torch.long)
        padded["attention_mask"] = torch.tensor(
            [pad_to(m, 0, max_len) for m in batch["attention_mask"]], dtype=torch.long
        )
        # Labels: pad with -100 (ignore index)
        padded["labels"] = torch.tensor([pad_to(l, -100, max_len) for l in labels_list], dtype=torch.long)

        # Teacher tensors
        t_pad_id = self.teacher_tokenizer.pad_token_id or 0
        padded["teacher_input_ids"] = torch.tensor(
            [pad_to(x, t_pad_id, max_len) for x in batch["teacher_input_ids"]], dtype=torch.long
        )
        padded["teacher_attention_mask"] = torch.tensor(
            [pad_to(m, 0, max_len) for m in batch["teacher_attention_mask"]], dtype=torch.long
        )
        return padded

collator = KDDataCollator(
    tokenizer=student_tokenizer,
    teacher_tokenizer=teacher_tokenizer,
    assistant_start_text=config["tokenizer"]["assistant_start_text"],
    assistant_end_text=config["tokenizer"]["assistant_end_text"],
    max_length=None,  # dynamic padding per batch
)

# -------------------------------
# Adaptation layer (student_h -> teacher_h), map true transformer layers
# Note: hidden_states[0] = embeddings, so we slice [1:]
# -------------------------------
class MultiLayerAdaptationLayer(nn.Module):
    def __init__(self, student_dim, teacher_dim, num_student_layers, num_teacher_layers, dtype=torch.bfloat16):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Linear(student_dim, teacher_dim, dtype=dtype) for _ in range(num_student_layers)
        ])
        self.dtype = dtype
        # map student layer index -> teacher layer index
        self.layer_mapping = {
            i: round(i * (num_teacher_layers - 1) / (num_student_layers - 1))
            for i in range(num_student_layers)
        }

    def forward(self, student_hidden_states_list: List[torch.Tensor], layer_indices=None) -> List[torch.Tensor]:
        # student_hidden_states_list excludes embeddings; length = num_student_layers or K
        # layer_indices: actual layer indices to use for projection selection
        if layer_indices is None:
            layer_indices = range(len(student_hidden_states_list))
        
        adapted = []
        for hs, idx in zip(student_hidden_states_list, layer_indices):
            if idx >= len(self.projections):
                break
            # Cast inputs to weight dtype (fp32) for stable computation
            weight_dtype = self.projections[idx].weight.dtype
            adapted.append(self.projections[idx](hs.to(weight_dtype)))
        return adapted

student_num_layers = getattr(student_model.config, "num_hidden_layers", None)
teacher_num_layers = getattr(teacher_model.config, "num_hidden_layers", None)
if student_num_layers is None or teacher_num_layers is None:
    raise ValueError("Could not infer num_hidden_layers from model configs.")

adaptor = MultiLayerAdaptationLayer(
    student_dim=student_model.config.hidden_size,
    teacher_dim=getattr(teacher_model.config, "hidden_size", student_model.config.hidden_size),
    num_student_layers=student_num_layers,
    num_teacher_layers=teacher_num_layers,
    dtype=torch.float32,  # Keep weights in fp32 for stability
)

# Move adaptor to same device (but keep fp32 weights)
device = next(student_model.parameters()).device
adaptor.to(device=device)

# -------------------------------
# Callback for gradient monitoring
# -------------------------------
class GradProbe(TrainerCallback):
    def _safe_format(self, val, fmt=".4f"):
        """Safely format values that might be tensors, floats, or strings"""
        if val == "N/A":
            return val
        try:
            if torch.is_tensor(val):
                return f"{float(val):{fmt}}"
            elif isinstance(val, (int, float)):
                return f"{val:{fmt}}"
            else:
                return str(val)
        except:
            return str(val)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 1 or state.global_step % 100 == 0:
            trainer = kwargs.get("trainer")
            if trainer and hasattr(trainer, "adaptor"):
                any_grad = any(p.grad is not None for p in trainer.adaptor.parameters())
                # Get latest logged values
                if trainer.log_history:
                    latest_log = trainer.log_history[-1]
                    kd_val = latest_log.get("loss_kd", "N/A")
                    ce_val = latest_log.get("loss_xe", "N/A")
                    alpha_val = latest_log.get("alpha", "N/A")
                    kd = self._safe_format(kd_val, ".4f")
                    ce = self._safe_format(ce_val, ".4f")
                    alpha = self._safe_format(alpha_val, ".3f")
                    print(f"[step {state.global_step}] adaptor_grads={any_grad} kd={kd} ce={ce} alpha={alpha}")
                else:
                    print(f"[step {state.global_step}] adaptor_grads={any_grad}")

# -------------------------------
# Trainer
# -------------------------------
class KDTrainer(SFTTrainer):
    def __init__(self, *args, teacher_model=None, adaptor=None, **kwargs):
        self.teacher_model = teacher_model
        self.adaptor = adaptor
        super().__init__(*args, **kwargs)

    def _no_weight_decay(self, name: str) -> bool:
        """Check if parameter should have no weight decay (biases, LayerNorm, etc.)"""
        return name.endswith(".bias") or "norm" in name.lower()

    # Make sure the adaptor is TRAINED by adding its params with proper weight decay
    def create_optimizer(self):
        if self.optimizer is None:
            # Separate student model params by weight decay
            student_groups = [
                {"params": [], "weight_decay": self.args.weight_decay},  # regular params
                {"params": [], "weight_decay": 0.0}                     # no decay params
            ]
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    group_idx = 1 if self._no_weight_decay(n) else 0
                    student_groups[group_idx]["params"].append(p)

            # Separate adaptor params by weight decay
            adaptor_groups = [
                {"params": [], "weight_decay": self.args.weight_decay},  # regular params
                {"params": [], "weight_decay": 0.0}                     # no decay params
            ]
            for n, p in self.adaptor.named_parameters():
                if p.requires_grad:
                    group_idx = 1 if self._no_weight_decay(n) else 0
                    adaptor_groups[group_idx]["params"].append(p)

            # Filter out empty groups
            all_groups = [g for g in student_groups + adaptor_groups if g["params"]]
            
            # Use fused AdamW on Hopper/Ampere GPUs for better performance
            use_fused = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
            self.optimizer = torch.optim.AdamW(
                all_groups, lr=self.args.learning_rate, fused=use_fused
            )
        return self.optimizer

    def _mean_pool(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Sequence-pooled representation: h [B,T,D], m [B,T] -> [B,D]"""
        m = m.unsqueeze(-1).to(h.dtype)  # [B, T, 1]
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    def _kd_loss_token_masked(
        self,
        student_h: torch.Tensor,            # [B, T, D_t]
        teacher_h: torch.Tensor,            # [B, T, D_t]
        s_mask: torch.Tensor,               # [B, T]
        t_mask: torch.Tensor,               # [B, T]
        loss_type: str = "cosine",
    ) -> torch.Tensor:
        # intersect masks and broadcast
        mask = (s_mask & t_mask).unsqueeze(-1).to(student_h.dtype)  # [B, T, 1]
        if loss_type == "mse":
            diff = (student_h - teacher_h) ** 2 * mask
            denom = mask.sum().clamp_min(1.0)
            return diff.sum() / denom
        elif loss_type == "huber":
            diff = F.huber_loss(student_h, teacher_h, reduction="none", delta=1.0) * mask
            denom = mask.sum().clamp_min(1.0)
            return diff.sum() / denom
        else:  # cosine
            s = F.normalize(student_h, p=2, dim=-1)
            t = F.normalize(teacher_h, p=2, dim=-1)
            cos = F.cosine_similarity(s, t, dim=-1)  # [B, T]
            cos = cos * mask.squeeze(-1)
            denom = mask.squeeze(-1).sum().clamp_min(1.0)
            return (1.0 - cos).sum() / denom

    def _kd_loss_pooled(
        self,
        student_h: torch.Tensor,            # [B, T, D_t]
        teacher_h: torch.Tensor,            # [B, T, D_t]
        s_mask: torch.Tensor,               # [B, T]
        t_mask: torch.Tensor,               # [B, T]
    ) -> torch.Tensor:
        """Sequence-pooled KD as fallback for tokenization mismatch"""
        s_vec = self._mean_pool(student_h, s_mask)  # [B, D_t]
        t_vec = self._mean_pool(teacher_h, t_mask)  # [B, D_t]
        s_norm = F.normalize(s_vec, p=2, dim=-1)
        t_norm = F.normalize(t_vec, p=2, dim=-1)
        cos_sim = F.cosine_similarity(s_norm, t_norm, dim=-1)  # [B]
        return (1.0 - cos_sim).mean()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Ensure adapter and teacher stay synced with model device (HF can move models during training)
        model_device = next(model.parameters()).device
        if next(self.adaptor.parameters()).device != model_device:
            self.adaptor.to(model_device)
        if next(self.teacher_model.parameters()).device != model_device:
            self.teacher_model.to(model_device)

        # Forward student
        student_out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
            output_hidden_states=True,
        )
        xe_loss = student_out.loss

        # Forward teacher (inference mode for better performance)
        with torch.inference_mode():
            teacher_out = self.teacher_model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
                output_hidden_states=True,
            )

        # Hidden states: drop embeddings at index 0
        s_hs = list(student_out.hidden_states)[1:]  # len = num_student_layers
        t_hs = list(teacher_out.hidden_states)[1:]  # len = num_teacher_layers

        # Optional: use only last-K layers for efficiency and stability
        last_k = config["distillation"]["last_k_layers"]
        if last_k is not None:
            K = min(last_k, len(s_hs), len(t_hs))
            s_hs = s_hs[-K:]
            t_hs = t_hs[-K:]
            # Pass the actual layer indices to use correct projections
            student_num_layers = getattr(model.config, "num_hidden_layers", len(s_hs) + K)
            s_idx = list(range(student_num_layers - K, student_num_layers))
            adapted_s = self.adaptor(s_hs, layer_indices=s_idx)
        else:
            # Use all layers
            adapted_s = self.adaptor(s_hs)
        s_mask = inputs["attention_mask"].bool()
        t_mask = inputs["teacher_attention_mask"].bool()

        # Get current training step for warm-up schedules
        step = getattr(self.state, "global_step", 0)
        
        kd_losses = []
        # Use pooled KD early for tokenization robustness, then switch to per-token
        use_pooled = (step < 1000) or config["distillation"]["use_pooled_kd"]
        
        if last_k is not None:
            # Direct mapping for last-K layers (i -> i)
            for i in range(len(adapted_s)):
                sh = adapted_s[i]            # [B, T, D_t]
                th = t_hs[i]                 # [B, T, D_t]
                # safety: trim to same T if any mismatch due to extreme truncation
                T = min(sh.size(1), th.size(1), s_mask.size(1), t_mask.size(1))
                
                if use_pooled:
                    # Use sequence-pooled KD for tokenization robustness
                    kd_losses.append(
                        self._kd_loss_pooled(
                            sh[:, :T, :], th[:, :T, :], s_mask[:, :T], t_mask[:, :T]
                        )
                    )
                else:
                    # Use per-token KD
                    kd_losses.append(
                        self._kd_loss_token_masked(
                            sh[:, :T, :], th[:, :T, :], s_mask[:, :T], t_mask[:, :T],
                            loss_type=config["distillation"]["loss_type"]
                        )
                    )
        else:
            # Use full layer mapping for all layers
            for s_idx, t_idx in self.adaptor.layer_mapping.items():
                sh = adapted_s[s_idx]            # [B, T, D_t]
                th = t_hs[t_idx]                 # [B, T, D_t]
                # safety: trim to same T if any mismatch due to extreme truncation
                T = min(sh.size(1), th.size(1), s_mask.size(1), t_mask.size(1))
                
                if use_pooled:
                    # Use sequence-pooled KD for tokenization robustness
                    kd_losses.append(
                        self._kd_loss_pooled(
                            sh[:, :T, :], th[:, :T, :], s_mask[:, :T], t_mask[:, :T]
                        )
                    )
                else:
                    # Use per-token KD
                    kd_losses.append(
                        self._kd_loss_token_masked(
                            sh[:, :T, :], th[:, :T, :], s_mask[:, :T], t_mask[:, :T],
                            loss_type=config["distillation"]["loss_type"]
                        )
                    )
        kd_loss = torch.stack(kd_losses).mean()

        # Guard against NaN XE loss (happens when all labels are -100)
        if (xe_loss is None) or torch.isnan(xe_loss):
            # batch has no valid supervised tokens → KD-only this step
            xe_loss = torch.zeros((), device=kd_loss.device, dtype=kd_loss.dtype)
            alpha_eff = 1.0
        else:
            # KD warm-up: ramp alpha from 0 → target over ~1000 steps
            alpha_max = config["distillation"]["alpha"]
            alpha_eff = float(min(alpha_max, alpha_max * (step / max(1, 1000))))

        total = alpha_eff * kd_loss + (1.0 - alpha_eff) * xe_loss
        # Safety net: replace any remaining NaNs with KD loss
        total = torch.nan_to_num(total, nan=kd_loss, posinf=kd_loss, neginf=kd_loss)

        # log for diagnostics
        # AFTER (safe: scalars)
        self.log({
            "loss_total": float(total.detach().mean().item()),
            "loss_xe": float(xe_loss.detach().mean().item()),
            "loss_kd": float(kd_loss.detach().mean().item()),
            "alpha": float(alpha_eff),
            "step": int(step),
            "use_pooled_kd": int(use_pooled),
        })


        return (total, student_out) if return_outputs else total

# -------------------------------
# Training args
# -------------------------------
training_args = TrainingArguments(
    **config["training"],
)

# -------------------------------
# Build Trainer
# -------------------------------
trainer = KDTrainer(
    model=student_model,
    train_dataset=dataset,
    args=training_args,
    data_collator=collator,
    teacher_model=teacher_model,
    adaptor=adaptor,
)

# Add gradient monitoring callback
trainer.add_callback(GradProbe())

# -------------------------------
# Train & Save
# -------------------------------
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

# Save student model + tokenizer
trainer.save_model(config["training"]["output_dir"])
student_tokenizer.save_pretrained(config["training"]["output_dir"])

# Save adaptor weights separately (optional)
os.makedirs(config["training"]["output_dir"], exist_ok=True)
torch.save(adaptor.state_dict(), os.path.join(config["training"]["output_dir"], "adaptation_layer.pth"))
print("✅ Training complete.")
