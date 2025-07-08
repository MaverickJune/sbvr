import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from awq import AutoAWQForCausalLM  # adjust import path if needed

def perf_timing(fn, x, device, runs=500):
    """
    Measure average execution time (μs) of fn(x) using time.perf_counter().
    """
    # Warm-up
    for _ in range(50):
        with torch.inference_mode(): _ = fn(x)
        if device.type == "cuda": torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(runs):
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode(): _ = fn(x)
        if device.type == "cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    # convert seconds to microseconds
    avg_us = (sum(times) / runs) * 1e6
    return avg_us


def measure_layer_components(model, layer_idx=0, view=False):  
    model.eval()
    device = next(model.parameters()).device

    # Tokenize & embed

    # Extract layer and its projections
    layer = model.model.layers[layer_idx]
    q_proj    = layer.self_attn.q_proj
    k_proj    = layer.self_attn.k_proj
    gate_proj = layer.mlp.gate_proj
    down_proj = layer.mlp.down_proj

    h = torch.randn(1, 1, layer.hidden_size, device="cuda:0", dtype=torch.float16)


    # Prepare dummy for down_proj
    hidden_dim = layer.mlp.intermediate_size  # usually 4 * hidden_size
    dummy_down = torch.randn(1, 1, hidden_dim, device=device, dtype=h.dtype)
    if view:
        h = h.view(1, -1)
        dummy_down = dummy_down.view(1, -1)

    # Measure each in μs
    results = {
        'q_proj_us':    perf_timing(q_proj,    h,           device),
        'k_proj_us':    perf_timing(k_proj,    h,           device),
        'gate_proj_us': perf_timing(gate_proj, h,           device),
        'down_proj_us': perf_timing(down_proj, dummy_down,  device),
    }
    return results

def measure_multiple_operators(model, layer_idx=0, view=False):  
    model.eval()
    device = next(model.parameters()).device


    # Extract layer and its projections
    layer = model.model.layers[layer_idx]

    def make_run_qkv_projs(layer):
        def run_qkv_projs(x):
            q_proj = layer.self_attn.q_proj
            k_proj = layer.self_attn.k_proj
            v_proj = layer.self_attn.v_proj
            with torch.inference_mode():
                q = q_proj(x)
                k = k_proj(x)
                v = v_proj(x)
            return torch.cat((q, k, v), dim=-1)
        return run_qkv_projs

    run_qkv = make_run_qkv_projs(layer)

    qkv = run_qkv
    mlp = layer.mlp

    h = torch.randn(1, 1, layer.hidden_size, device="cuda:0", dtype=torch.float16)
    
    if view:
        # qkv_gemv = SbvrGemvModule(
        #     q_proj          = layer.self_attn.q_proj,
        #     k_proj          = layer.self_attn.k_proj,
        #     v_proj          = layer.self_attn.v_proj,
        #     rtn_bits        = 7,
        #     rtn_group_size  = 128
        # )
        # qkv = qkv_gemv
        h = h.view(1, -1)

    # Measure each in μs
    results = {
        'qkv_linear': perf_timing(qkv, h, device),
        'mlp':       perf_timing(mlp,  h, device),
    }
    return results

def main():
    # 0) Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}\n")

    # 1) Define model names/paths
    base_model    = "meta-llama/Llama-3.2-1B"
    quant_path    = "Llama-3.2-1B-AWQ-gemv"
    layer_to_test = 0

    # 2) Load standard FP16 Torch model
    print("Loading standard FP16 model...")
    mdl_torch = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    ).to(device)

    # 3) Load AWQ-quantized model
    print("Loading AWQ-quantized model...")
    mdl_awq = AutoAWQForCausalLM.from_quantized(
        quant_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        trust_remote_code=True,
        fuse_layers=False
    ).to(device)

    # 4) Measure components on both models
    print("\nMeasuring Torch baseline projections...")
    metrics_torch = measure_layer_components(mdl_torch, layer_to_test)

    print("\nMeasuring AWQ projections...")
    metrics_awq   = measure_layer_components(mdl_awq, layer_to_test)

    # 5) Compare results in microseconds
    print("\nProjection comparison (µs):")
    header = f"{'Component':<15s}{'Torch':>12s}{'AWQ':>12s}{'Δ(µs)':>12s}"
    print(header)
    print("-" * len(header))
    for comp in ["q_proj_us", "k_proj_us", "gate_proj_us", "down_proj_us"]:
        t = metrics_torch[comp]
        a = metrics_awq[comp]
        delta = a - t
        print(f"{comp:<15s}{t:12.2f}{a:12.2f}{delta:12.2f}")


    # Measure multiple operators
    print("\nMeasuring multiple operators...")
    metrics_multi_torch = measure_multiple_operators(mdl_torch, layer_to_test)
    metrics_multi_awq   = measure_multiple_operators(mdl_awq, layer_to_test)

    # Compare multiple operators
    print("\nComparison of multiple operators (μs):")
    header_multi = f"{'Operator':<15s}{'Torch':>12s}{'AWQ':>12s}{'Δ(μs)':>12s}"
    print(header_multi)
    print('-' * len(header_multi))
    for op in ['qkv_linear', 'mlp']:
        t = metrics_multi_torch[op]
        s = metrics_multi_awq[op]
        delta = s - t
        print(f"{op:<15s}{t:12.2f}{s:12.2f}{delta:12.2f}")

if __name__ == "__main__":
    main()
