#!/usr/bin/env python
import time
import torch
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from awq.utils.utils import get_best_device

import os


def print_module_types(model):
    """
    Print each module’s name and its full class path (module + class name).
    """
    # Print the root model class
    root_cls = model.__class__
    print(f"\n[Model class] {root_cls.__module__}.{root_cls.__name__}\n")

    # Iterate all submodules
    for name, module in model.named_modules():
        cls = module.__class__
        print(f"{name:50s} → {cls.__module__}.{cls.__name__}")
    print("\n" + "="*80 + "\n")

def torch_backend():
        # --- Setup ---
    device = get_best_device()  # choose CUDA if available
    model_id = "meta-llama/Llama-3.2-1B"
    # model_id = "meta-llama/Llama-3.1-8B"  # Example for Llama-3.1-8B

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load full-precision model (FP16) on a single device
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.to(device)

    # --- 1) Print full model structure ---
    print_module_types(model)

    # --- 2) Prompt and inputs ---
    prompt = "Hello my name is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warm-up
    for _ in range(5):
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
            if device.startswith("cuda"):
                torch.cuda.synchronize()

    # --- 3) Measure end-to-end latency for 20 tokens ---
    latencies = []
    for _ in range(10):
        with torch.inference_mode():
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.time()
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.time()
        latencies.append(t1 - t0)

    avg_latency = sum(latencies) / len(latencies)

    # Decode and print
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n[prompt]   {prompt}")
    print(f"[response] {decoded}")
    print(f"[latency]  {avg_latency*1000:.2f} ms (end-to-end)")

def autoawq_backend():
    # --- Setup ---
    device = get_best_device()  # choose CUDA if available

    print(f"Using device: {device}")

    quant_path = "Llama-3.2-1B-AWQ-gemv"
    # quant_path = "Llama-3.1-8B-AWQ-gemm"  # Example for Llama-3.1-8B

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)

    # Load quantized model
    model = AutoAWQForCausalLM.from_quantized(
        quant_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",  # Automatically map to available devices
        trust_remote_code=True,
        fuse_layers=False
    )
    model.to(device)

    # --- 1) Print full model structure ---
    print_module_types(model)

    # --- 2) Prompt and inputs ---
    prompt = "Hello my name is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warm-up
    for _ in range(5):
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
            if device.startswith("cuda"):
                torch.cuda.synchronize()

    # --- 3) Measure end-to-end latency for 20 tokens ---
    # Timed runs
    latencies = []
    for _ in range(10):
        with torch.inference_mode():
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.time()
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.time()
        latencies.append(t1 - t0)

    avg_latency = sum(latencies) / len(latencies)

    # Decode and print
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n[prompt]   {prompt}")
    print(f"[response] {decoded}")
    print(f"[latency]  {avg_latency*1000:.2f} ms (end-to-end)")

if __name__ == "__main__":
    logging.set_verbosity_error()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    torch_backend()  # Run the full-precision model
    print("\n" + "="*80 + "\n")
    autoawq_backend()  # Run the AWQ-quantized model
    print("\n" + "="*80 + "\n")
    print("Inference complete. Compare the latency and outputs above.")
