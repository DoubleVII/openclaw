set -e

git clone https://github.com/DoubleVII/verl.git
git checkout gen_rm

#!/bin/bash

echo "Installing sglang..."
pip install "sglang[all]==0.5.2" --index-url $PYPI_MIRROR

echo "Installing vllm..."
pip install "vllm==0.11.0" --index-url $PYPI_MIRROR

echo "Installing basic packages..."
pip install "transformers[hf_xet]==4.57.6" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    "ray[default]" codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 \
    liger-kernel mathruler pytest py-spy pre-commit ruff tensorboard

echo "Installing additional packages..."
pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" \
    "pydantic>=2.9" "grpcio>=1.62.1"

echo "Installing FlashAttention..."
# export MAX_JOBS=8 # Avoid OOM/killed processes with limited RAM per core (common on shared clusters)
# export FLASH_ATTENTION_FORCE_BUILD=TRUE # forces flash-attn's setup.py to skip the pre-built wheel search and always compile from source
# export FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE # forces the new C++11 ABI (_GLIBCXX_USE_CXX11_ABI=1) during compilation
pip install flash-attn==2.8.3 --no-build-isolation

echo "Installing FlashInfer..."
pip install flashinfer-python==0.3.1 --index-url $PYPI_MIRROR

echo "Installing verl..."
pip install -e .

echo "Installing opencv..."
pip install opencv-python opencv-fixer

echo "Installing onnxscript (for TransformerEngine)..."
pip install "onnxscript==0.3.1"
pip install numpy==2.2.6

echo "Installing lingua-language-detector..."
pip install lingua-language-detector
