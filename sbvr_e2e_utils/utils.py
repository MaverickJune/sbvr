import torch
from transformers import LlamaForCausalLM
from utils.utils import cleanup_memory

@torch.inference_mode()
def get_partial_state(args):
    model_name = args.input_model
    ref = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu"
    ).eval()
    
    raw_state = ref.state_dict()
    filtered_state = {
        k: v for k, v in raw_state.items()
        if "self_attn" not in k and "mlp" not in k and "quantizer" not in k
    }
    
    del ref, raw_state
    cleanup_memory()
    
    return filtered_state
    