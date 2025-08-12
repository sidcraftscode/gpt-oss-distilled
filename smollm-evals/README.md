# SmolLM2
SmolLM2 is a family of compact language models available in three size: 135M, 360M, and 1.7B parameters. They are capable of solving a wide range of tasks while being lightweight enough to run on-device.

**New: Introducing [Smol-tools](#smol-tools) 🚀** 

<img src="https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/y45hIMNREW7w_XpHYB_0q.png" alt="Evaluation Results" width="600">

## Table of Contents
1. [Usage](#usage)
    - [Transformers](#transformers)
    - [Chat in TRL](#chat-in-trl)
    - [Local applications](#local-applications)
2. [Smol-tools](#smol-tools)
3. [Fine-tuning](#fine-tuning)
4. [Evaluation](#evaluation)

## Usage
Our most powerful model is `SmolLM2-1.7B-Instruct`, which you can use as an assistant with `transformers`, `trl`, or using quantized versions with tools like `llama.cpp`, `MLX`, and `transformers.js`. For lighter applications, you can also use the smaller models `SmolLM2-360M` and`SmolLM2-135M`, which are suitable for on-device usage and can be integrated similarly.
All available in this [collection](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9).

### Transformers
```bash
pip install transformers
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

device = "cuda" # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

messages = [{"role": "user", "content": "Write a 100-word article on 'Benefits of Open-Source in AI research"}]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
print(tokenizer.decode(outputs[0]))
```

### Chat in TRL
You can also use the TRL CLI to chat with the model from the terminal:
```bash
pip install trl
trl chat --model_name_or_path HuggingFaceTB/SmolLM2-1.7B-Instruct --device cpu
```

You can find more details on how to leverage the model for use cases such as text summarization, text rewriting and function calling in the model card: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct 

### Local applications
You can use the models locally with frameworks like `llama.cpp`, `MLX`, and `transformers.js`, which support SmolLM2. 
All models are available in this [collection](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9).

## Smol-tools
A collection of lightweight AI-powered tools built with LLaMA.cpp and small language models. These tools are designed to run locally on your machine without requiring expensive GPU resources.
Further instructions on how to use the tools can be found in the [smol-tools README](smol_tools/README.md).

## Fine-tuning
You can find an example script to finetune SmolLM2 using `TRL` and `PEFT` in the `finetune` folder.

## Evaluation
![image/png](https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/T-cHJVA7FBaI0cgDApzEj.png)

You can find more detailed evaluation of each model size in the model cards in this [collection](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9).
We use [lighteval](https://github.com/huggingface/lighteval) for all our evaluations, for more details refer to the [evaluation README](evaluation/README.md).
