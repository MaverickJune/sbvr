import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time
import torch
import sbvr
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy

out_dir = "data_mse"
os.makedirs(out_dir, exist_ok=True)

# General utility functions
def r_str(s):
    return "\033[91m" + str(s) + "\033[0m"
def g_str(s):
    return "\033[92m" + str(s) + "\033[0m"
def y_str(s):
    return "\033[93m" + str(s) + "\033[0m"
def b_str(s):
    return "\033[94m" + str(s) + "\033[0m"

def print_tensor(tensor, name="Tensor"):
    print(b_str(name) + ": " 
          + g_str("shape: ") + str(tensor.shape))
    print(tensor)

def get_errors(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError("Tensors must have the same shape")
    
    errors = tensor1 - tensor2
    mse = torch.mean(errors ** 2).item()
    max_error = torch.max(errors).item()
    min_error = torch.min(errors).item()
    std_dev = torch.std(errors).item()
    
    return errors, mse, max_error, min_error, std_dev

def print_errors(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"Tensors must have the same shape: "
                         f"{tensor1.shape} vs {tensor2.shape}")
    print(g_str("Tensor 1: ") +
          y_str("Mean: ") + f"{torch.mean(tensor1):.4e}" + ", " +
          y_str("ABS Mean: ") + f"{torch.mean(tensor1.abs()):.4e}" + ", " +
          y_str("Max: ") + f"{torch.max(tensor1):.4e}" + ", " +
          y_str("Min: ") + f"{torch.min(tensor1):.4e}" + ", " +
          y_str("Std. Dev.: ") + f"{torch.std(tensor1):.4e}")
    print(g_str("Tensor 2: ") +
          y_str("Mean: ") + f"{torch.mean(tensor2):.4e}" + ", " +
          y_str("ABS Mean: ") + f"{torch.mean(tensor2.abs()):.4e}" + ", " +
          y_str("Max: ") + f"{torch.max(tensor2):.4e}" + ", " +
          y_str("Min: ") + f"{torch.min(tensor2):.4e}" + ", " +
          y_str("Std. Dev.: ") + f"{torch.std(tensor2):.4e}")
    errors, mse, max_error, min_error, std_dev = get_errors(tensor1, tensor2)
    print(r_str("Errors:   ") + 
          y_str("MSE:  ") + f"{mse:.4e}" + ", " +
          y_str("ABS Mean: ") + f"{torch.mean(errors.abs()):.4e}" + ", " +
          y_str("Max: ") + f"{max_error:.4e}" + ", " +
          y_str("Min: ") + f"{min_error:.4e}" + ", " +
          y_str("Std. Dev.: ") + f"{std_dev:.4e}")
    
def load_or_create_tensor(name, shape, device):
    shape_str = "_".join(map(str, shape))
    file_path = f"{out_dir}/{name}_[{shape_str}].pt"
    if os.path.exists(file_path):
        return torch.load(file_path).to(device)
    else:
        tensor = torch.randn(shape, device=device, dtype=torch.float16) * 0.3
        torch.save(tensor, file_path)
        return tensor

def load_or_create_sbvr(data, name, shape, device, num_sums, verbose_level=0):
    shape_str = "_".join(map(str, shape))
    file_path = f"{out_dir}/sbvr_{num_sums}_{name}_[{shape_str}].pt"
    if os.path.exists(file_path):
        return sbvr.load(file_path, device=device, verbose_level=verbose_level)
    else:
        if data is None:
            raise ValueError(f"No cached SBVR found at {file_path} and no data tensor provided.")
        sbvr_tensor = sbvr.sbvr(data, encoder_config={"num_sums": num_sums}, 
                                device=device, verbose_level=verbose_level)
        sbvr_tensor.save(file_path)
        return sbvr_tensor
    
def create_sbvr(tensor, name, shape, device, num_sums, verbose_level=0):
    shape_str = "_".join(map(str, shape))
    file_path = f"{out_dir}/sbvr_{num_sums}_{name}_[{shape_str}].pt"
    sbvr_tensor = sbvr.sbvr(tensor, encoder_config={"num_sums": num_sums}, 
                            device=device, verbose_level=verbose_level)
    sbvr_tensor.save(file_path)
    return sbvr_tensor

# FP4 quantization functions
def float_to_fp4_e3m0(x):
    x_clamped = torch.clamp(x, -16.0, 16.0)
    sign = (x_clamped < 0).to(torch.uint8)

    x_abs = x_clamped.abs()
    is_zero = (x_abs == 0)
    x_abs = torch.clamp(x_abs, min=1e-8)

    exp_unbiased = torch.round(torch.log2(x_abs))
    exp_clamped = exp_unbiased.clamp(-3, 4)
    exp_q = (exp_clamped + 3).to(torch.uint8)

    fp4 = (sign << 3) | exp_q
    fp4 = torch.where(is_zero, torch.zeros_like(fp4), fp4)
    return fp4.to(torch.uint8)

def fp4_e3m0_to_float(fp4):
    sign = (fp4 >> 3) & 0b1
    exp_q = fp4 & 0b111
    is_zero = (exp_q == 0) & (sign == 0)

    exponent = exp_q.to(torch.int32) - 3
    value = 2.0 ** exponent
    value = torch.where(sign.bool(), -value, value)
    value = torch.where(is_zero, torch.zeros_like(value), value)
    return value

def float_to_fp4_e2m1(x):
    x_clamped = torch.clamp(x, -6.0, 6.0)
    sign = (x_clamped < 0).to(torch.uint8)
    x_abs = x_clamped.abs()

    is_zero = (x_abs == 0)
    x_abs = torch.clamp(x_abs, min=1e-8)

    exp_unbiased = torch.floor(torch.log2(x_abs))
    exp_clamped = exp_unbiased.clamp(-1, 2)
    exp_q = (exp_clamped + 1).to(torch.uint8)

    base = 2.0 ** exp_clamped
    mantissa = ((x_abs >= base * 1.25)).to(torch.uint8)

    fp4 = (sign << 3) | (exp_q << 1) | mantissa
    fp4 = torch.where(is_zero, torch.zeros_like(fp4), fp4)
    return fp4.to(torch.uint8)

def fp4_e2m1_to_float(fp4):
    sign = (fp4 >> 3) & 0b1
    exp_q = (fp4 >> 1) & 0b11
    mantissa = fp4 & 0b1
    is_zero = (exp_q == 0) & (mantissa == 0) & (sign == 0)

    exponent = exp_q.to(torch.int32) - 1
    base = 2.0 ** exponent
    value = base * (1.0 + 0.5 * mantissa)
    value = torch.where(sign.bool(), -value, value)
    value = torch.where(is_zero, torch.zeros_like(value), value)
    return value

def float_to_uint4_rtn(x):
    x_min = x.min()
    x_max = x.max()
    scale = (x_max - x_min) / 15.0
    scale = torch.clamp(scale, min=1e-8)  # prevent div by zero
    zero_point = torch.round(-x_min / scale).clamp(0, 15).to(torch.uint8)

    q = torch.round(x / scale) + zero_point
    q = q.clamp(0, 15).to(torch.uint8)
    return q, scale, zero_point

def uint4_rtn_to_float(q, scale, zero_point):
    return (q.to(torch.float32) - zero_point.to(torch.float32)) * scale

def float_to_int4_rtn(x):
    x_abs_max = x.abs().max()
    scale = x_abs_max / 7.0  # map to [-7, 7], reserve -8 for exact symmetry
    scale = torch.clamp(scale, min=1e-8)

    q = torch.round(x / scale)
    q = q.clamp(-8, 7).to(torch.int8)
    return q, scale

def int4_rtn_to_float(q, scale):
    return q.to(torch.float32) * scale

# Function for load partial weight of the actual model
@torch.inference_mode()
def load_partial_weight(model,
                        layer_num,
                        proj_name):
    # 1. Load the full weight of the model to CPU to avoid GPU OOM.
    loaded_model = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16, device_map="cpu"
    ).eval()
    state_dict = loaded_model.state_dict()

    # 2. Find all available linear module (proj) names and validate.
    available_projs = set()
    for key in state_dict:
        # Match keys like "model.layers.{N}.self_attn.q_proj.weight"
        parts = key.split(".")
        if "weight" in parts and any(p.endswith("_proj") or p in ("gate", "up", "down") for p in parts):
            for p in parts:
                if p.endswith("_proj") or p in ("gate", "up", "down"):
                    available_projs.add(p)

    if proj_name not in available_projs:
        raise ValueError(
            f"Projection '{proj_name}' not found. Available projections: {sorted(available_projs)}"
        )

    # Search for the weight matching layer_num and proj_name.
    target_key = None
    for key in state_dict:
        if f".{layer_num}." in key and f".{proj_name}.weight" in key:
            target_key = key
            break

    if target_key is None:
        raise ValueError(
            f"Could not find weight for layer {layer_num}, projection '{proj_name}' in state_dict."
        )

    # 3. Move the required weight to GPU and return.
    weight = state_dict[target_key].to("cuda:0")
    del loaded_model, state_dict
    return weight

# SBVR Accuracy test function
def sbvr_4bit_acc_test(
    weight,
    model_name,
    proj_name,
    layer_num,
):
    # NOTE: bvr_len is defaultly set to 128, in sbvr.core encoder_config. You can change it if you want
    device = "cuda:0" if torch.cuda.is_available() else None
    if device is None:
        print(r_str("No GPU available. Exiting test."))
        return
    
    target_weight_name = f"{model_name}_layer{layer_num}_{proj_name}"
    weight_e3m0 = fp4_e3m0_to_float(float_to_fp4_e3m0(weight))
    weight_e2m1 = fp4_e2m1_to_float(float_to_fp4_e2m1(weight))
    weight_rtn_uint4, scale_uint4, zp_uint4 = float_to_uint4_rtn(weight)
    weight_rtn_uint4_reconstructed = uint4_rtn_to_float(weight_rtn_uint4, scale_uint4, zp_uint4)
    weight_rtn_int4, scale_int4 = float_to_int4_rtn(weight)
    weight_rtn_int4_reconstructed = int4_rtn_to_float(weight_rtn_int4, scale_int4)
    weight_sbvr = load_or_create_sbvr(weight, target_weight_name, weight.shape, device, num_sums=4, verbose_level=1)
    weight_sbvr_reconstructed = weight_sbvr.decode()
    
    # Printout errors
    print(g_str(f"Comparing original weight with FP4 E3M0 quantized version:"))
    print_errors(weight, weight_e3m0)
    print(g_str(f"\nComparing original weight with FP4 E2M1 quantized version:"))
    print_errors(weight, weight_e2m1)
    print(g_str(f"\nComparing original weight with UINT4 RTN quantized version:"))
    print_errors(weight, weight_rtn_uint4_reconstructed)
    print(g_str(f"\nComparing original weight with INT4 RTN quantized version:"))
    print_errors(weight, weight_rtn_int4_reconstructed)
    print(g_str(f"\nComparing original weight with SBVR quantized version:"))
    print_errors(weight, weight_sbvr_reconstructed)

# Main entry
if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    layer_num = 0
    proj_name = "up_proj"
    
    # Load the partial weight from the actual model
    weight = load_partial_weight(model_name, layer_num, proj_name)
    sbvr_4bit_acc_test(weight, model_name.split("/")[-1], proj_name, layer_num)
    
    print(g_str("\nTest completed."))
    
