import torch
from sbvr import sbvr
from sbvr import load as sbvr_load
from sbvr.core import sbvrizer, load_sbvrizer, input_sbvr_mm_T
from sbvr.utils import print_errors, get_errors, r_str, y_str, b_str, g_str

import sys

@torch.inference_mode()
def test():
    device = "cuda:0"
    target_weight_sbvr_path = "./quantized_model/meta-llama_Llama-3.2-1B_4_16_16/sbvr_layer_0_self_attn.q_proj.module.pt"
    target_input_sbvrizer_path = "./input_profile/meta-llama_Llama-3.2-1B_4_16_16/per_state_encoding/0_k_proj.pt"
    target_data_path = "./input_profile/meta-llama_Llama-3.2-1B_4_16_16/layer_io/000.pt"

    # load the modules and the input data
    w_sbvr = sbvr_load(target_weight_sbvr_path, device=device)
    input_sbvrizer = load_sbvrizer(target_input_sbvrizer_path, device=device).to(torch.float16)

    batched_qkv_proj_input = torch.load(target_data_path, map_location=device)["input"]["k_proj"]
    batched_qkv_proj_input = batched_qkv_proj_input.reshape(-1, batched_qkv_proj_input.shape[-1])
    select_idx = torch.randint(0, batched_qkv_proj_input.shape[0], (1,))
    target_input = batched_qkv_proj_input[select_idx]

    # test the sbvrizer
    hidden_states = input_sbvrizer(target_input, mode=1)
    bvr, coeff_idx, coeff_set = input_sbvrizer.bvr, input_sbvrizer.coeff_idx, input_sbvrizer.coeff_set
    result_1 = input_sbvr_mm_T(bvr, coeff_idx, coeff_set, w_sbvr)

    decoded_weight = w_sbvr.decode()
    restored_input = input_sbvrizer.decode()
    result_2 = restored_input @ decoded_weight.T
    
    result_3 = target_input @ decoded_weight.T
    
    errors, mse, max_error, min_error, std_dev = get_errors(result_1, result_2)
    print(r_str("Errors:   ") + 
        y_str("MSE:  ") + f"{mse:.4e}" + ", " +
        y_str("ABS Mean: ") + f"{torch.mean(errors.abs()):.4e}" + ", " +
        y_str("Max: ") + f"{max_error:.4e}" + ", " +
        y_str("Min: ") + f"{min_error:.4e}" + ", " +
        y_str("Std. Dev.: ") + f"{std_dev:.4e}\n")
    
    errors, mse, max_error, min_error, std_dev = get_errors(result_1, result_3)
    print(r_str("Errors:   ") + 
        y_str("MSE:  ") + f"{mse:.4e}" + ", " +
        y_str("ABS Mean: ") + f"{torch.mean(errors.abs()):.4e}" + ", " +
        y_str("Max: ") + f"{max_error:.4e}" + ", " +
        y_str("Min: ") + f"{min_error:.4e}" + ", " +
        y_str("Std. Dev.: ") + f"{std_dev:.4e}\n")
    
    errors, mse, max_error, min_error, std_dev = get_errors(result_2, result_3)
    print(r_str("Errors:   ") + 
        y_str("MSE:  ") + f"{mse:.4e}" + ", " +
        y_str("ABS Mean: ") + f"{torch.mean(errors.abs()):.4e}" + ", " +
        y_str("Max: ") + f"{max_error:.4e}" + ", " +
        y_str("Min: ") + f"{min_error:.4e}" + ", " +
        y_str("Std. Dev.: ") + f"{std_dev:.4e}\n")
    
if __name__ == "__main__":
    test()