#!/bin/bash
#
set -e

VENV_NAME=venv

# CUDA
export CUDA_HOME=/usr/local/cuda-12
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Compiler
export CC=gcc-13
export CXX=g++-13
export CUDACXX=$CUDA_HOME/bin/nvcc

# For compiling extensions
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

uv venv $VENV_NAME --python 3.11

source $VENV_NAME/bin/activate

uv pip install setuptools
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

MAX_JOBS=4 uv pip install -r requirements.txt

# MAX_JOBS=4 uv pip install causal-conv1d --no-binary causal-conv1d --no-cache-dir --no-build-isolation

