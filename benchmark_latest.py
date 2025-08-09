import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

# Test prompts for evaluation
test_prompts = [
    "Explain quantum computing in simple terms",
    "Write a Python function to calculate fibonacci numbers",
    "What are the main causes of climate change?",
    "Describe the process of photosynthesis",
    "How does machine learning work?",
    "Explain the theory of relativity",
    "What is the difference between DNA and RNA?",
    "How do neural networks learn?"
]

def load_model(model_path, model_name):
    print(f"Loading {model_name}...")
    
    # Memory-efficient loading for large models
    if "gpt-oss-20b" in model_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa"  # Your preference
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def generate_response(model, tokenizer, prompt, model_name):
    # Format prompt
    if hasattr(tokenizer, 'apply_chat_template'):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"User: {prompt}\nAssistant:"
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Time the generation
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,  # Let models finish naturally
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generation_time = time.time() - start_time
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return response, generation_time, len(outputs[0]) - len(inputs.input_ids[0])

def benchmark_model(model_path, model_name):
    model, tokenizer = load_model(model_path, model_name)
    
    results = {
        "model_name": model_name,
        "model_path": model_path,
        "responses": [],
        "avg_generation_time": 0,
        "avg_tokens_generated": 0,
        "total_params": sum(p.numel() for p in model.parameters()),
        "model_size_gb": sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
    }
    
    total_time = 0
    total_tokens = 0
    
    for i, prompt in enumerate(test_prompts):
        print(f"  Testing prompt {i+1}/{len(test_prompts)}: {prompt[:50]}...")
        
        response, gen_time, tokens_generated = generate_response(model, tokenizer, prompt, model_name)
        
        total_time += gen_time
        total_tokens += tokens_generated
        
        results["responses"].append({
            "prompt": prompt,
            "response": response,
            "generation_time": gen_time,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_generated / gen_time if gen_time > 0 else 0
        })
        
        print(f"    Generated {tokens_generated} tokens in {gen_time:.2f}s ({tokens_generated/gen_time:.1f} tok/s)")
    
    results["avg_generation_time"] = total_time / len(test_prompts)
    results["avg_tokens_generated"] = total_tokens / len(test_prompts)
    results["avg_tokens_per_second"] = total_tokens / total_time if total_time > 0 else 0
    
    # Clean up memory
    del model, tokenizer
    torch.cuda.empty_cache()
    
    return results

def main():
    models_to_test = [
        {
            "path": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "name": "Original SmolLM2-1.7B"
        },
        {
            "path": "/workspace/distillation/gpt-oss-distilled/results/checkpoint-624",  # Updated to test 10k sample model
            "name": "Distilled SmolLM2-1.7B (10k samples)"
        },
        {
            "path": "openai/gpt-oss-20b",
            "name": "GPT-OSS-20B (Teacher)"
        }
    ]
    
    all_results = []
    
    for model_info in models_to_test:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_info['name']}")
        print(f"{'='*60}")
        
        try:
            results = benchmark_model(model_info["path"], model_info["name"])
            all_results.append(results)
            
            print(f"\n✅ {model_info['name']} Results:")
            print(f"   Parameters: {results['total_params']:,}")
            print(f"   Model Size: {results['model_size_gb']:.2f} GB")
            print(f"   Avg Generation Time: {results['avg_generation_time']:.2f}s")
            print(f"   Avg Tokens/Second: {results['avg_tokens_per_second']:.1f}")
            
        except Exception as e:
            print(f"❌ Error benchmarking {model_info['name']}: {e}")
    
    # Save detailed results
    with open("benchmark_results_latest.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    for result in all_results:
        print(f"{result['model_name']:<30} | "
              f"{result['total_params']/1e9:.1f}B params | "
              f"{result['model_size_gb']:.1f}GB | "
              f"{result['avg_tokens_per_second']:.1f} tok/s")

if __name__ == "__main__":
    main()