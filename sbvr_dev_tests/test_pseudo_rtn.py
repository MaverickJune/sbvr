import torch
import os
import sys
from sbvr_e2e_utils.utils import r_str, y_str, b_str, g_str, get_errors, print_errors
from tqdm import tqdm

class PseudoRTN:
    def __init__(self, model_name, w_bits, device="cuda:0", rtn_bits=5):
        self.model_name = self.model_name_formatter(model_name, w_bits)
        self.input_data_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            ), f"input_profile/{self.model_name}/layer_io"
        )
        self.device = device
        self.rtn_bits = rtn_bits
        self.rtn_board = self.get_rtn_board(rtn_bits).to(device)
    
    def model_name_formatter(self, model_name, w_bits, a_bits=16, kv_bits=16):
        model_name = model_name.replace("/", "_")
        return f"{model_name}_{w_bits}_{a_bits}_{kv_bits}"
    
    def convert_layer_to_input_filename(self, layer_idx):
        return f"{layer_idx:03d}.pt"
    
    def get_rtn_board(self, rtn_bits):
        max_num = 2 ** rtn_bits - 1
        rtn_board = torch.arange(-max_num, max_num + 1, 1, dtype=torch.float16, device=self.device)
        return rtn_board
    
    @torch.inference_mode()
    def apply_pseudo_rtn(self, x):
        if x.dim() != 2 or x.shape[0] != 1:
            raise ValueError(f"x must be a 2D tensor with shape (1, n), but got {x.shape}")
        x = x.to(self.device)
        x = x.reshape(-1, 1)
        a = x.min()
        b = x.max()
        s = max(abs(a), abs(b)) / 31
        scaled_rtn_board = self.rtn_board * s
        selected_indices = torch.argmin(torch.abs(x - scaled_rtn_board), dim=-1)
        
        x_q = scaled_rtn_board[selected_indices]
        x_q = x_q.reshape(1, -1)
        
        return x_q, selected_indices.to(torch.uint8)
    
    def load_input_data(self, layer_idx):
        input_path = os.path.join(self.input_data_path, self.convert_layer_to_input_filename(layer_idx))
        return torch.load(input_path, map_location=self.device)
    
    def apply_rtn_to_input(self, x):
        if x.dim() != 2:
            raise ValueError("x must be a 2D tensor")
        x_orig_shape = x.shape
        
        x = x.reshape(-1, 128)
        x_q = torch.zeros_like(x)
        
        for i in tqdm(range(x.shape[0]), desc="applying RTN quantization", ncols=80):
            xt = x[i].view(1, -1)
            xt_q, _ = self.apply_pseudo_rtn(xt)
            x_q[i] = xt_q
            
        return x_q.reshape(x_orig_shape)
        
    def test_rtn_acc_module(self, layer_idx, module_name, n_samples=50):
        module_name_to_sbvrizer_dict = {
            "q_proj": "k_proj",
            "k_proj": "k_proj",
            "v_proj": "k_proj",
            "o_proj": "o_proj",
            "up_proj": "gate_proj",
            "gate_proj": "gate_proj",
            "down_proj": "down_proj"
        }
        io_indicator = "output" if module_name in ["v_proj"] else "input"
        input_data = self.load_input_data(layer_idx)
        batched_proj_input = input_data[io_indicator][module_name_to_sbvrizer_dict[module_name]]
        batched_proj_input = batched_proj_input.reshape(-1, batched_proj_input.shape[-1])
        select_idx = torch.randint(0, batched_proj_input.shape[0], (n_samples,))
        batched_proj_input = batched_proj_input[select_idx]
        
        x_q = self.apply_rtn_to_input(batched_proj_input)
        print_errors(batched_proj_input, x_q)
        
        
if __name__ == "__main__":
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    W_BITS = 4
    device = "cuda:0"
    
    def randn_test():
        rtn_worker = PseudoRTN(MODEL_NAME, W_BITS, device, rtn_bits=5)
        x = torch.randn(1, 128, dtype=torch.float16, device="cuda:0")
        x_q, selected_indices = rtn_worker.apply_pseudo_rtn(x)
        print_errors(x, x_q)
        
    def proj_test(layer_idx, module_name):
        rtn_worker = PseudoRTN(MODEL_NAME, W_BITS, device, rtn_bits=5)
        rtn_worker.test_rtn_acc_module(layer_idx=layer_idx, module_name=module_name)
        
    proj_test(layer_idx=0, module_name="q_proj")
    
    
   