import time
import torch
import os

from paper_eval_package.cudagraph_utils import _make_cg_runner
from sbvr_e2e import load_sbvr_llama_model

def print_module_types(model):
    """
    Print each module's name and its full class path (module + class name).
    """
    root_cls = model.__class__
    print(f"\n[Model class] {root_cls.__module__}.{root_cls.__name__}\n")

    for name, module in model.named_modules():
        cls = module.__class__
        print(f"{name:50s} → {cls.__module__}.{cls.__name__}")
    print("\n" + "="*80 + "\n")

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

def measure_ablation_latency(model, layer_idx=0, use_cudagraph=False, target_op=None):
    model.eval()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this function.")
    device = torch.device("cuda:0")
    
    layer = model.model.layers[layer_idx]
    hidden_size = model.config.hidden_size
    
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
    
    if target_op is None:
        raise ValueError("target_op must be specified for ablation latency measurement.")
    
    TARGET_OPERATORS = {
        "q_proj": layer.self_attn.q_proj,
        "k_proj": layer.self_attn.k_proj,
        "v_proj": layer.self_attn.v_proj,
        "o_proj": layer.self_attn.o_proj,
        "mlp_up_proj": layer.mlp.up_proj,
        "mlp_gate_proj": layer.mlp.gate_proj,
        "mlp_down_proj": layer.mlp.down_proj,
        "mlp": layer.mlp,
        "qkv_projs": make_run_qkv_projs(layer)
    }
    if target_op not in TARGET_OPERATORS:
        raise ValueError(f"Unsupported target_op: {target_op}. Supported operators: {list(TARGET_OPERATORS.keys())}")
    target_operation = TARGET_OPERATORS[target_op]
    
    # Prepare input
    h = torch.randn(1, 1, hidden_size, device="cuda:0", dtype=torch.float16)
    
    # Apply cudagraph if needed
    if use_cudagraph:
        target_operation = _make_cg_runner(target_operation, h)
    
    # Measure latency
    results = {target_op: perf_timing(target_operation, h, device)}
    
    return results
    
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}\n")
    layer_to_test = 0
    
    # Change this part to load the SBVR-quantized model instead of the original model
    root_sbvr_path = "/home/wjbang/workspace/sbvr_quantized_models/meta-llama_Llama-3.1-8B-Instruct_4_16_16_w_rotate"
    input_model = "meta-llama/Llama-3.1-8B-Instruct"
    
    print("Loading SBVR-quantized model...")
    model = load_sbvr_llama_model(
        root_sbvr_path=root_sbvr_path,
        input_model=input_model
    )
    print_module_types(model)
    
    
