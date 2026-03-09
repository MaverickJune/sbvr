import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import time

from paper_eval_package.cudagraph_utils import _make_cg_runner

# DEFAULT_SEED = 42
# def set_deterministic_seed(seed: int = DEFAULT_SEED):
#     """Fix all random sources so that every run produces identical values."""
#     random.seed(seed)
#     torch.manual_seed(seed)          # covers both CPU and CUDA
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     # Set CUBLAS workspace config for deterministic cuBLAS calls
#     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
#     print(f"[seed] All random sources fixed with seed={seed}")
# set_deterministic_seed()

from sbvr_e2e import load_sbvr_llama_model, load_sbvr_qwen2_model
from sbvr import _sbvr_input_transfrom

def perf_timing(fn, x, device, runs=500, warmup=50):
    for _ in range(warmup):
        with torch.inference_mode():
            _ = fn(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = fn(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return (sum(times) / runs) * 1e6

def print_module_types(model):
    root_cls = model.__class__
    print(f"\n[Model class] {root_cls.__module__}.{root_cls.__name__}\n")
    for name, module in model.named_modules():
        cls = module.__class__
        print(f"{name:50s} → {cls.__module__}.{cls.__name__}")
    print("\n" + "="*80 + "\n")

@torch.inference_mode()
def measure_sbvr_operator_latency(
    model,
    operator_name,
    layer_idx=0,
    warmup_runs=10,
    runs=50,
    use_cudagraph=True,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This function requires a CUDA-enabled device.")
    device = torch.device("cuda:0")
    layer = model.model.layers[layer_idx]
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    
    rtn_bits = int(model.config.rtn_bits)
    rtn_group_size = int(model.config.rtn_group_size)
    
    # qkvo_projections
    def q_proj_fn(x):
        out_bvr, scales = _sbvr_input_transfrom(
            x, nRTN=rtn_bits, group_size=rtn_group_size
        )
        return layer.self_attn.q_proj.d_forward(out_bvr=out_bvr, scales=scales)

    def k_proj_fn(x):
        out_bvr, scales = _sbvr_input_transfrom(
            x, nRTN=rtn_bits, group_size=rtn_group_size
        )
        return layer.self_attn.k_proj.d_forward(out_bvr=out_bvr, scales=scales)
    
    def v_proj_fn(x):
        out_bvr, scales = _sbvr_input_transfrom(
            x, nRTN=rtn_bits, group_size=rtn_group_size
        )
        return layer.self_attn.v_proj.d_forward(out_bvr=out_bvr, scales=scales)
    
    def o_proj_fn(x):
        out_bvr, scales = _sbvr_input_transfrom(
            x, nRTN=rtn_bits, group_size=rtn_group_size
        )
        return layer.self_attn.o_proj.d_forward(out_bvr=out_bvr, scales=scales)
    
    # mlp_projections
    def gate_proj_fn(x):
        out_bvr, scales = _sbvr_input_transfrom(
            x, nRTN=rtn_bits, group_size=rtn_group_size
        )
        return layer.mlp.gate_proj.d_forward(out_bvr=out_bvr, scales=scales)
    
    def up_proj_fn(x):
        out_bvr, scales = _sbvr_input_transfrom(
            x, nRTN=rtn_bits, group_size=rtn_group_size
        )
        return layer.mlp.up_proj.d_forward(out_bvr=out_bvr, scales=scales)
    
    def down_proj_fn(x):
        return layer.mlp.down_proj.d_forward(x)
    
    def make_qkv_proj(layer):
        def run_qkv_projs(x):
            with torch.inference_mode():
                q_out = q_proj_fn(x)
                k_out = k_proj_fn(x)
                v_out = v_proj_fn(x)
            return torch.cat((q_out, k_out, v_out), dim=-1)
        return run_qkv_projs
    
    OPERATOR_DICT = {
        "q_proj": q_proj_fn,
        "k_proj": k_proj_fn,
        "v_proj": v_proj_fn,
        "o_proj": o_proj_fn,
        "gate_proj": gate_proj_fn,
        "up_proj": up_proj_fn,
        "down_proj": down_proj_fn,
        "qkv_proj": make_qkv_proj(layer),
        "mlp": layer.mlp
    }
    if operator_name not in OPERATOR_DICT:
        raise ValueError(f"Unsupported operator_name '{operator_name}'. Supported operators: {list(OPERATOR_DICT.keys())}")
    operator_fn = OPERATOR_DICT[operator_name]
    
    # Generate dummy input
    DUMMY_INPUT_DICT = {
        "q_proj": torch.randn(1, hidden_size, device=device, dtype=torch.float16),
        "k_proj": torch.randn(1, hidden_size, device=device, dtype=torch.float16),
        "v_proj": torch.randn(1, hidden_size, device=device, dtype=torch.float16),
        "o_proj": torch.randn(1, hidden_size, device=device, dtype=torch.float16),
        "gate_proj": torch.randn(1, hidden_size, device=device, dtype=torch.float16),
        "up_proj": torch.randn(1, hidden_size, device=device, dtype=torch.float16),
        "down_proj": torch.randn(1, intermediate_size, device=device, dtype=torch.float16),
        "qkv_proj": torch.randn(1, hidden_size, device=device, dtype=torch.float16),
        "mlp": torch.randn(1, 1, hidden_size, device=device, dtype=torch.float16)
    }
    dummy_input = DUMMY_INPUT_DICT[operator_name]
    
    # Apply cudagraph wrapper if requested
    if use_cudagraph:
        operator_fn = _make_cg_runner(operator_fn, dummy_input)
        print(f"[cudagraph] Enabled cudagraph for operator '{operator_name}'")
    else:
        print(f"[cudagraph] Running without cudagraph for operator '{operator_name}'")
    
    result = {
        f"{operator_name}": perf_timing(operator_fn, dummy_input, device, runs=runs, warmup=warmup_runs)
    }
    
    return result

def main():
    # root_sbvr_path list
    # "/home/wjbang/workspace/sbvr_quantized_models/Qwen_Qwen2.5_7B_Instruct_4_16_16_wo_rotate"
    # "/home/wjbang/workspace/sbvr_quantized_models/meta-llama_Llama-3.1-8B-Instruct_4_16_16_w_rotate"
    
    # input_model list
    # "Qwen/Qwen2.5-7B-Instruct"
    # "meta-llama/Llama-3.1-8B-Instruct"
    
    root_sbvr_path = "/home/wjbang/workspace/sbvr_quantized_models/meta-llama_Llama-3.1-8B-Instruct_4_16_16_w_rotate"
    input_model = "meta-llama/Llama-3.1-8B-Instruct"
    if input_model == "meta-llama/Llama-3.1-8B-Instruct":
        model = load_sbvr_llama_model(root_sbvr_path=root_sbvr_path, input_model=input_model)
    elif input_model == "Qwen/Qwen2.5-7B-Instruct":
        model = load_sbvr_qwen2_model(root_sbvr_path=root_sbvr_path, input_model=input_model)
    else:
        raise ValueError(f"Unsupported input_model '{input_model}'. Supported models: 'Qwen/Qwen2.5-7B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct'")
    model.config.enable_kernel_nvtx = False
    model.config.kernel_profile_layer = 0
    
    # Target operator for ablation
    operator_name = "down_proj"
    
    # Print model info
    print_module_types(model)
    
    # Measure latency
    print("\nMeasuring target operation latency...")
    result = measure_sbvr_operator_latency(model, operator_name, layer_idx=0, warmup_runs=10, runs=50, use_cudagraph=True)
    header = f"{'Operator':<15s}{'SBVR':>14s}"
    print(header)
    print('-' * len(header))
    print(f"{operator_name:<15s}{result[operator_name]:>14.2f} μs")
    
if __name__ == "__main__":
    main()