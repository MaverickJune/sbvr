import torch
from transformers import LlamaForCausalLM # , Qwen3ForCausalLM
from utils.utils import cleanup_memory

def r_str(s):
    return "\033[91m" + str(s) + "\033[0m"
def g_str(s):
    return "\033[92m" + str(s) + "\033[0m"
def y_str(s):
    return "\033[93m" + str(s) + "\033[0m"
def b_str(s):
    return "\033[94m" + str(s) + "\033[0m"

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
          y_str("Std. Dev.: ") + f"{std_dev:.4e}\n")

@torch.inference_mode()
def get_partial_state(args):
    model_name = args.input_model

    if "llama" in model_name.lower():
        print("Getting Partial State for Llama Model...")
        ref = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cpu"
        ).eval()
        raw_state = ref.state_dict()
        filtered_state = {
            k: v for k, v in raw_state.items()
            if "self_attn" not in k and "mlp" not in k and "quantizer" not in k
        }
    elif "qwen3" in model_name.lower():
        print("Getting Partial State for Qwen3 Model...")
        ref = Qwen3ForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="cpu"
        ).eval()
        
        raw_state = ref.state_dict()

        filtered_state = {
            k: v for k, v in raw_state.items()
            if (
                "mlp" not in k
                and "quantizer" not in k
                and (
                    "self_attn" not in k
                    or "q_norm" in k
                    or "k_norm" in k
                    or k.endswith("bias")
                )
            )
        }
    
    del ref, raw_state
    cleanup_memory()
    
    return filtered_state
    