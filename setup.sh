#!/bin/bash
# Install PyTorch
pip install torch==2.6.0

# Install wheel, packaging, and ninja
pip install packaging==24.2 ninja==1.11.1.3

# Install flash-attn and deepspeed
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install  deepspeed==0.16.3

# Install requirements from requirements.txt
pip install -r requirements.txt
