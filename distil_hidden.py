import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorWithPadding
from accelerate import Accelerator
import yaml
from dataclasses import dataclass
from typing import Dict, List, Any

# Set tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class CustomDataCollator:
    """Custom data collator that preserves teacher inputs along with student inputs"""
    tokenizer: AutoTokenizer
    padding: bool = True
    max_length: int = None
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract all the keys we need to preserve
        batch = {}
        
        # Standard keys for student model
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in features[0]:
                batch[key] = [feature[key] for feature in features]
        
        # Teacher-specific keys
        for key in ["teacher_input_ids", "teacher_attention_mask"]:
            if key in features[0]:
                batch[key] = [feature[key] for feature in features]
        
        # Convert to tensors and pad if needed
        for key, values in batch.items():
            if isinstance(values[0], list):
                # Pad sequences
                max_len = max(len(v) for v in values) if self.max_length is None else self.max_length
                padded_values = []
                for v in values:
                    if len(v) < max_len:
                        if key in ["input_ids", "teacher_input_ids"]:
                            pad_value = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                        elif key in ["attention_mask", "teacher_attention_mask"]:
                            pad_value = 0
                        elif key == "labels":
                            pad_value = -100
                        else:
                            pad_value = 0
                        v = v + [pad_value] * (max_len - len(v))
                    elif len(v) > max_len:
                        v = v[:max_len]
                    padded_values.append(v)
                batch[key] = torch.tensor(padded_values)
            else:
                batch[key] = torch.tensor(values)
        
        return batch

# Configuration
config = {
    "project_name": "distil-multilayer",
    "dataset": {
        "name": "mlabonne/FineTome-100k",
        "split": "train",
        "num_samples": 10, # Small test run to verify script works, then increase for full training
        "seed": 42
    },
    "models": {
        "teacher": "openai/gpt-oss-20b",
        "student": "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    },
    "tokenizer": {
        "max_length": 4096,
        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    },
    "training": {
        "output_dir": "./results",
        "num_train_epochs": 1,  # Single epoch often sufficient for distillation with large datasets
        "per_device_train_batch_size": 4,  # Increased for H100 - can go higher
        "gradient_accumulation_steps": 4,  # Reduced since batch size increased  
        "save_steps": 1000,
        "logging_steps": 2,
        "save_total_limit": 2,
        "learning_rate": 1e-5,  # Lower LR for distillation to avoid catastrophic forgetting
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,  # Shorter warmup for distillation
        "lr_scheduler_type": "linear",
        "resume_from_checkpoint": None,
        "fp16": False,
        "bf16": True,
        "max_grad_norm": 1.0,
        "group_by_length": False,
        "gradient_checkpointing": True,  # Essential for memory efficiency on H100
        "dataloader_pin_memory": True,  # H100 optimization
        "dataloader_num_workers": 8,  # Parallel data loading
        "remove_unused_columns": False,  # Required for distillation
    },
    "distillation": {
        "temperature": 2.0,  # Not used for hidden state distillation, kept for compatibility
        "alpha": 0.7,  # Increased focus on distillation loss for better knowledge transfer
        "loss_type": "cosine"  # Options: "mse", "cosine", "huber" - cosine often works best for representations
    },
    "model_config": {
        "use_flash_attention": False,  # Disabled due to import issues
        "torch_compile": True,  # H100 benefits significantly from torch.compile
        "use_cache": False,  # Disable KV cache during training for memory savings
    },
    # H100 Advanced Optimization Notes:
    # - For maximum utilization, add DeepSpeed: "deepspeed": "./deepspeed_configs/zero2.json"
    # - Can increase per_device_train_batch_size to 8-16 with proper memory management
    # - Consider sequence length tuning: shorter sequences = larger batch sizes
}

# Set up environment
os.environ['WANDB_PROJECT'] = config["project_name"]
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

accelerator = Accelerator()
device = accelerator.device

# Load and preprocess dataset
dataset = load_dataset(config["dataset"]["name"], split=config["dataset"]["split"])
if config["dataset"].get("num_samples"):
    dataset = dataset.select(range(config["dataset"]["num_samples"]))
dataset = dataset.shuffle(seed=config["dataset"]["seed"])

# Load tokenizers
teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])
student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])

# Set padding tokens
if teacher_tokenizer.pad_token is None:
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
if student_tokenizer.pad_token is None:
    student_tokenizer.pad_token = student_tokenizer.eos_token

# Apply chat template to student tokenizer
student_tokenizer.chat_template = config["tokenizer"]["chat_template"]

def prepare_dataset(example):
    system = "You are a helpful assistant chatbot."
    conversations = example['conversations']
    
    message = [{"role": "system", "content": system}]
    
    for conversation in conversations:
        if conversation.get('from') == 'human':
            message.append({"role": "user", "content": conversation.get('value', '')})
        elif conversation.get('from') == 'gpt':
            message.append({"role": "assistant", "content": conversation.get('value', '')})
    
    student_text = student_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    teacher_text = teacher_tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    
    student_encodings = student_tokenizer(student_text, truncation=True, max_length=config["tokenizer"]["max_length"], padding='max_length')
    teacher_encodings = teacher_tokenizer(teacher_text, truncation=True, max_length=config["tokenizer"]["max_length"], padding='max_length')

    return {
        "input_ids": student_encodings["input_ids"],
        "attention_mask": student_encodings["attention_mask"],
        "labels": student_encodings["input_ids"].copy(),  # For language modeling, labels are same as input_ids
        "teacher_input_ids": teacher_encodings["input_ids"],
        "teacher_attention_mask": teacher_encodings["attention_mask"],
    }

# Preprocess and tokenize the dataset
print("Preprocessing and tokenizing dataset...")
original_columns = dataset.column_names
dataset = dataset.map(prepare_dataset, remove_columns=original_columns)

print("Dataset preparation complete. Loading models...")

# Load models with configurable flash attention
model_kwargs = {"torch_dtype": torch.bfloat16 if config["training"]["bf16"] else (torch.float16 if config["training"]["fp16"] else torch.float32)}
if config["model_config"]["use_flash_attention"]:
    model_kwargs["attn_implementation"] = "flash_attention_2"

# Disable cache during training for memory efficiency
if not config["model_config"]["use_cache"]:
    model_kwargs["use_cache"] = False

teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], **model_kwargs).to(device)
student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], **model_kwargs).to(device)

# Apply torch.compile for H100 optimization (if enabled)
if config["model_config"]["torch_compile"]:
    print("🚀 Applying torch.compile optimization for H100...")
    student_model = torch.compile(student_model, mode="max-autotune")
    teacher_model = torch.compile(teacher_model, mode="max-autotune")

class MultiLayerAdaptationLayer(torch.nn.Module):
    def __init__(self, student_dim, teacher_dim, num_student_layers, num_teacher_layers, dtype=torch.bfloat16):
        super().__init__()
        self.projections = torch.nn.ModuleList([
            torch.nn.Linear(student_dim, teacher_dim, dtype=dtype)
            for _ in range(num_student_layers)
        ])
        self.layer_mapping = self.create_layer_mapping(num_student_layers, num_teacher_layers)
        self.dtype = dtype

    def create_layer_mapping(self, num_student_layers, num_teacher_layers):
        return {
            i: round(i * (num_teacher_layers - 1) / (num_student_layers - 1))
            for i in range(num_student_layers)
        }

    def forward(self, student_hidden_states):
        adapted_hidden_states = []
        for i, hidden_state in enumerate(student_hidden_states):
            if i >= len(self.projections):
                break
            adapted_hidden_states.append(self.projections[i](hidden_state.to(self.dtype)))
        return adapted_hidden_states

adaptation_layer = MultiLayerAdaptationLayer(
    student_model.config.hidden_size,
    teacher_model.config.hidden_size,
    student_model.config.num_hidden_layers,
    teacher_model.config.num_hidden_layers,
    dtype=torch.bfloat16
).to(device)

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        self.remove_unused_columns = kwargs.pop('remove_unused_columns', None)
        self.max_seq_length = kwargs.pop('max_seq_length', 1024)  # Use pop to remove from kwargs
        self.tokenizer = kwargs.pop('tokenizer', None)  # Remove tokenizer from kwargs
        self.packing = kwargs.pop('packing', False)  # Remove packing from kwargs
        super(CustomSFTTrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

        labels = inputs["labels"]

        student_outputs = model(**student_inputs, labels=labels, output_hidden_states=True)
        
        original_loss = student_outputs.loss

        self.teacher_model = self.teacher_model
        teacher_model = self.teacher_model.module if hasattr(self.teacher_model, 'module') else self.teacher_model

        with torch.no_grad():
            teacher_inputs = {
                "input_ids": inputs["teacher_input_ids"],
                "attention_mask": inputs["teacher_attention_mask"],
            }
            
            teacher_outputs = teacher_model(**teacher_inputs, output_hidden_states=True)

        custom_loss = self.distillation_loss(student_outputs, teacher_outputs, inputs, original_loss)
        return (custom_loss, student_outputs) if return_outputs else custom_loss
        
    def distillation_loss(self, student_outputs, teacher_outputs, inputs, original_loss):
        student_hidden_states = student_outputs.hidden_states
        teacher_hidden_states = teacher_outputs.hidden_states

        self.adaptation_layer = self.adaptation_layer.to(student_hidden_states[0].device)
        adapted_student_hidden_states = self.adaptation_layer(student_hidden_states)

        total_loss_kd = 0
        for student_hidden, teacher_idx in self.adaptation_layer.layer_mapping.items():
            teacher_hidden = teacher_hidden_states[teacher_idx]
            
            if adapted_student_hidden_states[student_hidden].shape != teacher_hidden.shape:
                raise ValueError(f"Shape mismatch: student {adapted_student_hidden_states[student_hidden].shape} vs teacher {teacher_hidden.shape}")

            # Option 1: MSE Loss (most common for hidden state distillation)
            if config["distillation"].get("loss_type", "mse") == "mse":
                loss_kd = F.mse_loss(
                    adapted_student_hidden_states[student_hidden], 
                    teacher_hidden,
                    reduction='mean'
                )
            
            # Option 2: Cosine Similarity Loss (often better for representations)
            elif config["distillation"]["loss_type"] == "cosine":
                # Normalize hidden states
                student_norm = F.normalize(adapted_student_hidden_states[student_hidden], p=2, dim=-1)
                teacher_norm = F.normalize(teacher_hidden, p=2, dim=-1)
                
                # Cosine similarity (we want to maximize it, so minimize negative)
                cosine_sim = F.cosine_similarity(student_norm, teacher_norm, dim=-1)
                loss_kd = (1 - cosine_sim).mean()  # Convert to loss (0 = perfect match)
            
            # Option 3: Huber Loss (robust to outliers)
            elif config["distillation"]["loss_type"] == "huber":
                loss_kd = F.huber_loss(
                    adapted_student_hidden_states[student_hidden],
                    teacher_hidden,
                    reduction='mean',
                    delta=1.0
                )
            
            else:
                raise ValueError(f"Unknown loss_type: {config['distillation']['loss_type']}")

            total_loss_kd += loss_kd

        avg_loss_kd = total_loss_kd / len(self.adaptation_layer.layer_mapping)
        
        # Scale the loss appropriately
        hidden_dim = adapted_student_hidden_states[0].size(-1)
        scaled_loss_kd = avg_loss_kd / (hidden_dim ** 0.5)  # Scale by sqrt(hidden_dim) for stability

        total_loss = config["distillation"]["alpha"] * scaled_loss_kd + (1 - config["distillation"]["alpha"]) * original_loss
        return total_loss

# Training arguments
training_arguments = TrainingArguments(
    **config["training"],
)

# Create custom data collator
custom_data_collator = CustomDataCollator(
    tokenizer=student_tokenizer,
    max_length=config["tokenizer"]["max_length"]
)

# Create the custom SFT Trainer
trainer = CustomSFTTrainer(
    model=student_model,
    train_dataset=dataset,
    max_seq_length=config["tokenizer"]["max_length"],
    tokenizer=student_tokenizer,
    args=training_arguments,
    packing=config["training"].get("packing", False),
    data_collator=custom_data_collator,
)

# Add these attributes to the trainer
trainer.teacher_model = teacher_model
trainer.adaptation_layer = adaptation_layer
trainer.student_tokenizer = student_tokenizer
trainer.teacher_tokenizer = teacher_tokenizer

# Prepare for distributed training
trainer = accelerator.prepare(trainer)

# Train the model
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

# Save the final model
trainer.save_model(config["training"]["output_dir"])

# Save the adaptation layer
torch.save(adaptation_layer.state_dict(), os.path.join(config["training"]["output_dir"], "adaptation_layer.pth"))
