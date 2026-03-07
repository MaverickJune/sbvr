import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import random

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Load the sbvr model instantiation functions
from sbvr_e2e import load_sbvr_qwen2_model
from paper_eval_package.cudagraph_utils import attach_cudagraph_generate_triton
from paper_eval_package.latency_benchmark import evaluate_latency

# ── Deterministic seeding ──────────────────────────────────────────────
DEFAULT_SEED = 42

def set_deterministic_seed(seed: int = DEFAULT_SEED):
    """Fix all random sources so that every run produces identical values."""
    random.seed(seed)
    torch.manual_seed(seed)          # covers both CPU and CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set CUBLAS workspace config for deterministic cuBLAS calls
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(f"[seed] All random sources fixed with seed={seed}")

def sbvr_e2e_latency_qwen(apply_cudagraph: bool = True):
    # Set deterministic seed for reproducibility
    set_deterministic_seed(DEFAULT_SEED)
    
    # Load the SBVR model and tokenizer
    root_sbvr_path = "/home/wjbang/workspace/sbvr_quantized_models/Qwen_Qwen2.5_7B_Instruct_4_16_16_wo_rotate"
    input_model = "Qwen/Qwen2.5-7B-Instruct"
    sbvr_model = load_sbvr_qwen2_model(root_sbvr_path, input_model) 
    tokenizer = AutoTokenizer.from_pretrained(input_model)
    print("SBVR model loaded successfully!")
    
    # Set configs for graph attach and latency evaluation
    device = "cuda:0"
    model_dtype = torch.float16
    
    # Attach the cuGraph generate function to the model
    if apply_cudagraph:
        attach_cudagraph_generate_triton(sbvr_model, tokenizer, device=device, dtype=model_dtype)
    else:
        print("Skipping cuGraph attachment as apply_cudagraph is set to False.")
    
    # Evaluate latency
    evaluate_latency(model=sbvr_model, tokenizer=tokenizer)
    
if __name__ == "__main__":
    sbvr_e2e_latency_qwen(apply_cudagraph=True)