import time
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from eval_utils.modelling_llama_sbvr import LlamaForSbvrLM, SbvrGemvModule
from sbvr_e2e_utils.utils import get_partial_state

# Reuse the microsecond timing helper

def perf_timing(fn, x, device, runs=1000):
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
    h_raw = h
    
    if view:
        qkv_gemv = SbvrGemvModule(
            q_proj          = layer.self_attn.q_proj,
            k_proj          = layer.self_attn.k_proj,
            v_proj          = layer.self_attn.v_proj,
            rtn_bits        = 7,
            rtn_group_size  = 128
        )
        qkv = qkv_gemv
        h = h.view(1, -1)

    # Measure each in μs
    results = {
        'qkv_linear': perf_timing(qkv, h, device),
        'mlp':       perf_timing(mlp,  h_raw, device),
    }
    return results


def main():
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}\n")

    # Model paths
    base_model      = "meta-llama/Llama-3.2-3B"
    root_sbvr_path  = "/home/nxclab/wonjun/sbvr/quantized_model/meta-llama_Llama-3.2-3B_4_16_16"
    layer_to_test   = 0


    # SBVR model init
    print("\nInitializing SBVR model...")
    # Placeholder args structure
    class Args: pass
    args = Args()
    args.root_sbvr_path = root_sbvr_path
    args.weight_bvr_len = 128
    args.weight_num_sums = 4
    args.rtn_group_size = 128
    args.rtn_bits = 7
    args.input_model = base_model

    filtered_state = get_partial_state(args)
    sbvr_state_dict = {
        "weight_bvr_len":  args.weight_bvr_len,
        "weight_num_sums": args.weight_num_sums,
        "rtn_group_size":  args.rtn_group_size,
        "rtn_bits":        args.rtn_bits,
    }
    config = AutoConfig.from_pretrained(base_model)

    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True

    model_sbvr = LlamaForSbvrLM(config=config, sbvr_state_dict=sbvr_state_dict)
    missing, unexpected = model_sbvr.load_state_dict(filtered_state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys: {unexpected}")
    if process_word_embeddings:
        model_sbvr.lm_head.weight.data = model_sbvr.model.embed_tokens.weight.data.clone()
    model_sbvr.load_sbvr_weights(root_sbvr_path)
    model_sbvr.convert_model_dtype(torch.float16)
    model_sbvr.preprocess_model()
    model_sbvr = model_sbvr.to(device)
    model_sbvr.eval()

    # Measure
    print("\nMeasuring Torch GEMV projections...")

    print("Measuring SBVR GEMV projections...")
    metrics_sbvr = measure_layer_components(model_sbvr, layer_to_test, True)

    # Compare
    print("\nComparison (μs):")
    header = f"{'Component':<15s}{'SBVR':>12s}"
    print(header)
    print('-' * len(header))
    for comp in ['q_proj_us','k_proj_us','gate_proj_us','down_proj_us']:
        s = metrics_sbvr[comp]
        print(f"{comp:<15s}{s:12.2f}")


    # Measure multiple operators
    print("\nMeasuring multiple operators...")
    metrics_multi_sbvr = measure_multiple_operators(model_sbvr, layer_to_test, True)

    # Compare multiple operators
    print("\nComparison of multiple operators (μs):")
    header_multi = f"{'Operator':<15s}{'SBVR':>12s}"
    print(header_multi)
    print('-' * len(header_multi))
    for op in ['qkv_linear', 'mlp']:
        s = metrics_multi_sbvr[op]
        print(f"{op:<15s}{s:12.2f}")
    

    

if __name__ == "__main__":
    
    main()
