#!/bin/bash
export CUDA_HOME=/usr/local/cuda-12.4
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=8

pip install flash-attn --no-build-isolation --verbose
