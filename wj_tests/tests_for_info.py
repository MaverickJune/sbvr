import argparse
import copy
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Tests
def test_model_instantiate():
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    model_dtype = torch.float16
    device = "cuda:0"
    
    # Instantiate the model
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=model_dtype, device_map=device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        model = None
        
    # Check the model type
    class_name = model.__class__.__name__
    full_class_path = f"{model.__class__.__module__}.{class_name}"

    print(f"Model class name: {class_name}")
    print(f"Model full class path: {full_class_path}")
    print(f"type(model): {type(model)}")
    

if __name__ == "__main__":
    test_model_instantiate()