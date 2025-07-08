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
    Evaluate inference latency:
      - TTFT: Time To First Token
      - TBT: average time per subsequent token
      - 20-token end-to-end latency
    """
    device = next(model.parameters()).device
    print(f"Using device: {device}")

    # 1) Print full model structure
    print_module_types(model)

    # 2) Prepare the prompt
    prompt = "Hello my name is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 3) Warm-up (5 runs)
    for _ in range(5):
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # 4) Measure TTFT (average over 10 runs)
    ttft_runs = 10
    ttft_latencies = []
    for _ in range(ttft_runs):
        with torch.inference_mode():
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
        ttft_latencies.append(t1 - t0)
    avg_ttft = sum(ttft_latencies) / len(ttft_latencies)
    avg_ttft_ms = avg_ttft * 1000

    # 5) Measure 20-token end-to-end latency (average over 10 runs)
    full_runs = 10
    full_latencies = []
    for _ in range(full_runs):
        with torch.inference_mode():
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
        full_latencies.append(t1 - t0)
    avg_full = sum(full_latencies) / len(full_latencies)
    avg_full_ms = avg_full * 1000

    # 6) Calculate TBT: (full latency - TTFT) / (number of additional tokens)
    #    Average time per token for the 19 tokens after the first
    avg_tbt_ms = (avg_full - avg_ttft) * 1000 / (20 - 1)

    # 7) Print results
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n[prompt]   {prompt}")
    print(f"[response] {decoded}")
    print(f"[TTFT]     {avg_ttft_ms:.2f} ms (to first token)")
    print(f"[TBT]      {avg_tbt_ms:.2f} ms (per additional token)")
    print(f"[latency]  {avg_full_ms:.2f} ms (20-token end-to-end)")

    return {
        "ttft_ms": avg_ttft_ms,
        "tbt_ms": avg_tbt_ms,
        "full_latency_ms": avg_full_ms,
        "prompt": prompt,
        "response": decoded
    }