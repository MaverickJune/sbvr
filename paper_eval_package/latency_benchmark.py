import time
import torch
import os

def print_module_types(model):
    """
    Print each module's name and its full class path (module + class name).
    """
    # Print the root model class
    root_cls = model.__class__
    print(f"\n[Model class] {root_cls.__module__}.{root_cls.__name__}\n")

    # Iterate all submodules
    for name, module in model.named_modules():
        cls = module.__class__
        print(f"{name:50s} → {cls.__module__}.{cls.__name__}")
    print("\n" + "="*80 + "\n")

def evaluate_latency(model, tokenizer):
    """
    Evaluate inference latency - follows autoawq_backend() logic exactly.
    """
    # Set CUDA device to 0 (same as autoawq_inference.py)
    
    # Get device (model should already be on cuda:0)
    device = next(model.parameters()).device
    print(f"Using device: {device}")

    # --- 1) Print full model structure ---
    print_module_types(model)

    # --- 2) Prompt and inputs ---
    prompt = "Hello my name is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warm-up (exactly 5 runs like autoawq_backend)
    for _ in range(10):
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # --- 3) Measure end-to-end latency for 20 tokens ---
    # Timed runs (exactly 10 runs like autoawq_backend)
    latencies = []
    for _ in range(10):
        with torch.inference_mode():
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
        latencies.append(t1 - t0)

    avg_latency = sum(latencies) / len(latencies)

    # Decode and print (same format as autoawq_backend)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n[prompt]   {prompt}")
    print(f"[response] {decoded}")
    print(f"[latency]  {avg_latency*1000:.2f} ms (end-to-end)")

    return {
        "avg_latency_ms": avg_latency * 1000,
        "prompt": prompt,
        "response": decoded
    }