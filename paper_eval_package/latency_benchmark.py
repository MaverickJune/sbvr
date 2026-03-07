import time
import torch
import os

DEFAULT_LATENCY_PROMPTS = [
    "Hello my name is",
    "Write a short description of a rainy afternoon in Seoul.",
    "Explain what matrix multiplication does in simple terms.",
    "Give me three ideas for a healthy breakfast.",
    "What are the key steps in debugging CUDA kernels?",
    "Summarize the plot of a sci-fi story in two sentences.",
    "List practical tips for reducing inference latency.",
    "Describe the concept of quantization for neural networks.",
    "How do attention mechanisms work in transformers?",
    "Give a short itinerary for a one-day trip in Busan.",
    "Explain why caching helps autoregressive decoding.",
    "Write a polite email asking for a project deadline extension.",
    "What are common causes of GPU out-of-memory errors?",
    "Describe differences between fp16 and int8 inference.",
    "Create a short motivational message for engineers.",
    "What is the benefit of batched inference workloads?",
    "Suggest a simple weekly workout plan for beginners.",
    "Explain the meaning of TTFT and TBT in LLM serving.",
    "Provide three ideas for improving code readability.",
    "Write a concise summary of climate change impacts.",
    "How can we profile CUDA kernels effectively?",
    "Describe a fantasy city floating above the clouds.",
    "What is the role of layer normalization in transformers?",
    "Give a checklist for launching an ML experiment.",
    "Explain the difference between prefill and decode stages.",
    "Write a short product description for wireless earbuds.",
    "What are pros and cons of static cache generation?",
    "Suggest five interview questions for backend engineers.",
    "Describe an efficient study routine for learning math.",
    "How do we choose batch size for latency experiments?",
    "Write a haiku about parallel computing.",
    "Give three practical tips for writing maintainable tests.",
]


def print_module_types(model):
    """
    Print each module's name and its full class path (module + class name).
    """
    root_cls = model.__class__
    print(f"\n[Model class] {root_cls.__module__}.{root_cls.__name__}\n")

    for name, module in model.named_modules():
        cls = module.__class__
        print(f"{name:50s} → {cls.__module__}.{cls.__name__}")
    print("\n" + "=" * 80 + "\n")


def _select_prompts(batch_size, prompts=None):
    src = prompts if prompts is not None else DEFAULT_LATENCY_PROMPTS
    if len(src) < batch_size:
        raise ValueError(
            f"Not enough prompts for batch_size={batch_size}. "
            f"Given={len(src)}, required>={batch_size}."
        )
    return list(src[:batch_size])


def evaluate_latency(model, tokenizer, batch_size=1, prompts=None):
    """
    Evaluate inference latency:
      - TTFT: Time To First Token
      - TBT: average time per subsequent token
      - 20-token end-to-end latency

    Timing is measured per batched generate call (wall-time).
    """
    device = next(model.parameters()).device
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")

    print_module_types(model)

    selected_prompts = _select_prompts(batch_size=batch_size, prompts=prompts)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(model, "generation_config") and model.generation_config is not None:
        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    inputs = tokenizer(
        selected_prompts,
        return_tensors="pt",
        padding=True,
    ).to(device)

    ttft_new_tokens = 1
    full_new_tokens = 20

    # 1) Warm-up and path priming
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=full_new_tokens, do_sample=False)
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(5):
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=full_new_tokens, do_sample=False)
            if device.type == "cuda":
                torch.cuda.synchronize()

    for _ in range(2):
        with torch.inference_mode():
            _ = model.generate(**inputs, max_new_tokens=ttft_new_tokens, do_sample=False)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # 2) TTFT
    ttft_runs = 10
    ttft_latencies = []
    for _ in range(ttft_runs):
        with torch.inference_mode():
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.generate(**inputs, max_new_tokens=ttft_new_tokens, do_sample=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
        ttft_latencies.append(t1 - t0)
    avg_ttft = sum(ttft_latencies) / len(ttft_latencies)
    avg_ttft_ms = avg_ttft * 1000

    # 3) Full latency
    full_runs = 10
    full_latencies = []
    for _ in range(full_runs):
        with torch.inference_mode():
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = model.generate(**inputs, max_new_tokens=full_new_tokens, do_sample=False)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
        full_latencies.append(t1 - t0)
    avg_full = sum(full_latencies) / len(full_latencies)
    avg_full_ms = avg_full * 1000

    # 4) TBT
    avg_tbt_ms = (avg_full - avg_ttft) * 1000 / (full_new_tokens - ttft_new_tokens)

    # 5) Decode and print all batch responses
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for i, (prompt, resp) in enumerate(zip(selected_prompts, decoded)):
        print(f"\n[prompt:{i}]   {prompt}")
        print(f"[response:{i}] {resp}")
    print(f"[TTFT]          {avg_ttft_ms:.2f} ms (to first token, batched call)")
    print(f"[TBT]           {avg_tbt_ms:.2f} ms (per additional token, batched call)")
    print(f"[latency]       {avg_full_ms:.2f} ms ({full_new_tokens}-token end-to-end, batched call)")

    return {
        "ttft_ms": avg_ttft_ms,
        "tbt_ms": avg_tbt_ms,
        "full_latency_ms": avg_full_ms,
        "prompts": selected_prompts,
        "responses": decoded,
        "batch_size": batch_size,
    }

# # Legacy codes 
# def print_module_types(model):
#     """
#     Print each module's name and its full class path (module + class name).
#     """
#     # Print the root model class
#     root_cls = model.__class__
#     print(f"\n[Model class] {root_cls.__module__}.{root_cls.__name__}\n")

#     # Iterate all submodules
#     for name, module in model.named_modules():
#         cls = module.__class__
#         print(f"{name:50s} → {cls.__module__}.{cls.__name__}")
#     print("\n" + "="*80 + "\n")

# def evaluate_latency(model, tokenizer):
#     """
#     Evaluate inference latency:
#       - TTFT: Time To First Token
#       - TBT: average time per subsequent token
#       - 20-token end-to-end latency
#     """
#     device = next(model.parameters()).device
#     print(f"Using device: {device}")

#     # 1) Print full model structure
#     print_module_types(model)

#     # 2) Prepare the prompt
#     prompt = "Hello my name is"
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)

#     # 3) Warm-up (5 runs)
#     for _ in range(5):
#         with torch.inference_mode():
#             _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
#             if device.type == "cuda":
#                 torch.cuda.synchronize()

#     # 4) Measure TTFT (average over 10 runs)
#     ttft_runs = 10
#     ttft_latencies = []
#     for _ in range(ttft_runs):
#         with torch.inference_mode():
#             if device.type == "cuda":
#                 torch.cuda.synchronize()
#             t0 = time.time()
#             _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
#             if device.type == "cuda":
#                 torch.cuda.synchronize()
#             t1 = time.time()
#         ttft_latencies.append(t1 - t0)
#     avg_ttft = sum(ttft_latencies) / len(ttft_latencies)
#     avg_ttft_ms = avg_ttft * 1000

#     # 5) Measure 20-token end-to-end latency (average over 10 runs)
#     full_runs = 10
#     full_latencies = []
#     for _ in range(full_runs):
#         with torch.inference_mode():
#             if device.type == "cuda":
#                 torch.cuda.synchronize()
#             t0 = time.time()
#             outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
#             if device.type == "cuda":
#                 torch.cuda.synchronize()
#             t1 = time.time()
#         full_latencies.append(t1 - t0)
#     avg_full = sum(full_latencies) / len(full_latencies)
#     avg_full_ms = avg_full * 1000

#     # 6) Calculate TBT: (full latency - TTFT) / (number of additional tokens)
#     #    Average time per token for the 19 tokens after the first
#     avg_tbt_ms = (avg_full - avg_ttft) * 1000 / (20 - 1)

#     # 7) Print results
#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(f"\n[prompt]   {prompt}")
#     print(f"[response] {decoded}")
#     print(f"[TTFT]     {avg_ttft_ms:.2f} ms (to first token)")
#     print(f"[TBT]      {avg_tbt_ms:.2f} ms (per additional token)")
#     print(f"[latency]  {avg_full_ms:.2f} ms (20-token end-to-end)")

#     return {
#         "ttft_ms": avg_ttft_ms,
#         "tbt_ms": avg_tbt_ms,
#         "full_latency_ms": avg_full_ms,
#         "prompt": prompt,
#         "response": decoded
#     }