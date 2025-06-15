import torch
import os
import sys
from sbvr_e2e_utils.utils import r_str, y_str, b_str, g_str, get_errors, print_errors
from sbvr import _sbvr_input_transfrom, _rtn_sbvr_1xtN_mm_T
from sbvr import load as sbvr_load
from utils.utils import cleanup_memory
from tqdm import tqdm

class PseudoRTN:
    def __init__(self, model_name, w_bits, device="cuda:0", rtn_bits=6):
        self.model_name = self._model_name_formatter(model_name, w_bits)
        self.input_data_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            ), f"input_profile/{self.model_name}/layer_io"
        )
        self.weight_sbvr_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            ), f"quantized_model/{self.model_name}"
        )
        self.weight_sbvr = None
        self.device = device
        self.rtn_bits = rtn_bits
        self.rtn_max_num = 2 ** (rtn_bits - 1) - 1
        self.rtn_board = self._get_rtn_board(rtn_bits).to(device)
        self.pseudo_rtn_pivots = self._get_rtn_pivots()
    
    def _model_name_formatter(self, model_name, w_bits, a_bits=16, kv_bits=16):
        model_name = model_name.replace("/", "_")
        return f"{model_name}_{w_bits}_{a_bits}_{kv_bits}"
    
    def _convert_layer_to_input_filename(self, layer_idx):
        return f"{layer_idx:03d}.pt"
    
    def _set_weight_sbvr(self, layer_idx, module_name = None):
        module_name_list = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
        if layer_idx == -1 or module_name == None:
            raise ValueError("layer_idx and module_name must be provided")
        if module_name not in module_name_list:
            raise ValueError(f"module_name must be one of {module_name_list}")
        
        module_indicator = "self_attn" if module_name in ["q_proj", "k_proj", "v_proj", "o_proj"] else "mlp"
        
        weight_sbvr_path = os.path.join(self.weight_sbvr_path, f"sbvr_layer_{layer_idx}_{module_indicator}.{module_name}.module.pt")
        self.weight_sbvr = sbvr_load(weight_sbvr_path, device=self.device, apply_dtype_conv=True, target_dtype=torch.float16)

    def _get_rtn_board(self, rtn_bits):
        rtn_board = torch.arange(-self.rtn_max_num, self.rtn_max_num + 1, 1, dtype=torch.float16, device=self.device)
        return rtn_board
    
    def _get_rtn_pivots(self):
        pseudo_rtn_pivots = [2 ** i for i in range(0, self.rtn_bits, 1)] + [-self.rtn_max_num]
        return torch.tensor(pseudo_rtn_pivots, dtype=torch.int16, device=self.device)
    
    def _get_module_sample_from_raw_input(self, input_data, module_name, n_samples=1, dtype=torch.float16):
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
        batched_proj_input = input_data[io_indicator][module_name_to_sbvrizer_dict[module_name]]
        batched_proj_input = batched_proj_input.reshape(-1, batched_proj_input.shape[-1])
        select_idx = torch.randint(0, batched_proj_input.shape[0], (n_samples,))
        batched_proj_input = batched_proj_input[select_idx].to(self.device).to(dtype)
        
        return batched_proj_input
        
    @torch.inference_mode()
    def _apply_pseudo_rtn(self, x, return_scale=False):
        if x.dim() != 2 or x.shape[0] != 1:
            raise ValueError(f"x must be a 2D tensor with shape (1, n), but got {x.shape}")
        x = x.to(self.device)
        x = x.reshape(-1, 1)
        a = x.min()
        b = x.max()
        s = max(abs(a), abs(b)) / self.rtn_max_num
        scaled_rtn_board = self.rtn_board * s
        selected_indices = torch.argmin(torch.abs(x - scaled_rtn_board), dim=-1)
        
        x_q = scaled_rtn_board[selected_indices]
        x_q = x_q.reshape(1, -1)
        if return_scale:
            return x_q, selected_indices.to(torch.uint8), s
        else:
            return x_q, selected_indices.to(torch.uint8)
    
    def _load_input_data(self, layer_idx, device=None):
        input_path = os.path.join(self.input_data_path, self._convert_layer_to_input_filename(layer_idx))
        if device is not None:
            return torch.load(input_path, map_location=device)
        else:
            return torch.load(input_path, map_location=self.device)
    
    def _apply_rtn_to_input(self, x, return_scale=False):
        if x.dim() != 2:
            raise ValueError("x must be a 2D tensor")
        x_orig_shape = x.shape
        
        x = x.reshape(-1, 128)
        x_q = torch.zeros_like(x)
        
        if return_scale:
            scales = torch.zeros(x.shape[0], device=x.device)
        
        for i in tqdm(range(x.shape[0]), desc="applying RTN quantization", ncols=80):
            xt = x[i].view(1, -1)
            if not return_scale:
                xt_q, _ = self._apply_pseudo_rtn(xt)
            else:
                xt_q, _, s = self._apply_pseudo_rtn(xt, return_scale=True)
                scales[i] = s
            x_q[i] = xt_q
        
        if return_scale:
            return x_q.reshape(x_orig_shape), scales
        else:
            return x_q.reshape(x_orig_shape)
        
    def _uint32_to_bitvector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert a tensor of uint32 values to their 32-bit binary representation.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of arbitrary shape whose dtype is torch.uint32.

        Returns
        -------
        torch.Tensor
            Tensor of shape (*x.shape, 32) and dtype torch.uint8
            containing the bit-vectors.  Index 0 is the least-significant bit.
        """
        if x.dtype != torch.uint32:
            raise TypeError("Input tensor must have dtype=torch.uint32")

        # use an int64 shift mask – rshift for int64 *is* implemented on CUDA
        shifts = torch.arange(32, device=x.device, dtype=torch.int64)

        # promote to int64 (safe: all uint32 values fit in positive int64 range)
        x64 = x.to(torch.int64)

        # logical right-shift, mask the LSB, cast to uint8
        bits = ((x64.unsqueeze(-1) >> shifts) & 1).to(torch.uint8)
        return bits
        
    def _decode_sbvr_rtn(self, input_data, out_bvr=None, scales=None):
        '''
        out_bvr: low_bit_pos -> high_bit_pos
        '''
        if out_bvr is None or scales is None:
            raise ValueError("out_bvr and scales must be provided")
        _nRTN = self.rtn_bits + 1
        restored_input = torch.zeros_like(input_data)
        
        out_bvr = out_bvr.flatten().reshape(-1, _nRTN)
        for i in tqdm(range(out_bvr.shape[0]), desc="decoding sbvr rtn", ncols=80):
            if i % 4 == 0:
                s = scales[i//4]
                scaled_pivots = self.pseudo_rtn_pivots * s
                scaled_pivots = scaled_pivots.to(out_bvr.device)
            xt = out_bvr[i].view(-1)
            xt = self._uint32_to_bitvector(xt)
            scaled_bitvecs = xt * scaled_pivots.unsqueeze(-1)      
            pr = scaled_bitvecs.sum(dim=0)
            restored_input[:, i*32 : (i + 1)*32] = pr.view(1, -1)
            
        return restored_input
    
    @torch.inference_mode()
    def _apply_rtn_sbvr_mm_T(self, out_bvr, scales):
        if self.weight_sbvr is None:
            raise ValueError("weight_sbvr is not set")
        bias = self.weight_sbvr._get_dummy_bias()
        return _rtn_sbvr_1xtN_mm_T(
            l_bvr = out_bvr,
            l_scales = scales,
            r_bvr = self.weight_sbvr.bvr,
            r_coeff_idx = self.weight_sbvr.coeff_idx,
            r_coeff_cache = self.weight_sbvr.coeff_cache,
            bias = bias,
            nRTN = self.rtn_bits
        )
        
    def test_rtn_acc_module(self, layer_idx, module_name, n_samples=50, input_data=None):
        if input_data is None:
            input_data = self._load_input_data(layer_idx)
        batched_proj_input = self._get_module_sample_from_raw_input(input_data, module_name, n_samples)
        
        x_q = self._apply_rtn_to_input(batched_proj_input)
        print_errors(batched_proj_input, x_q)
        
    def test_rtn_acc_module_all_layers(self, n_layers=16):
        module_name_list = ["k_proj", "o_proj", "gate_proj", "down_proj"]
        for layer_idx in range(n_layers):
            print(b_str(f"testing layer {layer_idx}...\n"))
            input_data = self._load_input_data(layer_idx, device="cpu")
            for module_name in module_name_list:
                print(f"{layer_idx}_{module_name}")
                self.test_rtn_acc_module(layer_idx=layer_idx, module_name=module_name, input_data=input_data)
            cleanup_memory()
            
    def test_sbvr_input_transform_rtn(self, layer_idx, module_name):
        input_data = self._load_input_data(layer_idx, device="cpu")
        proj_input = self._get_module_sample_from_raw_input(input_data, module_name, n_samples=1, dtype=torch.float16)
        rtn_encoded_input = self._apply_rtn_to_input(proj_input)
        
        # sbvr input transform
        out_bvr, scales = _sbvr_input_transfrom(proj_input, nRTN=self.rtn_bits, group_size=128)
        
        restored_proj_input = self._decode_sbvr_rtn(proj_input, out_bvr, scales)
        print_errors(proj_input, restored_proj_input)
        print_errors(rtn_encoded_input, restored_proj_input)
        
    def test_rtn_sbvr_mm_T(self, layer_idx, module_name):
        if self.weight_sbvr is None:
            self._set_weight_sbvr(layer_idx, module_name)
        
        input_data = self._load_input_data(layer_idx, device="cpu")
        proj_input = self._get_module_sample_from_raw_input(input_data, module_name, n_samples=1, dtype=torch.float16)
        
        out_bvr, scales = _sbvr_input_transfrom(proj_input, nRTN=self.rtn_bits, group_size=128)
        out = self._apply_rtn_sbvr_mm_T(out_bvr, scales)
        
        # emulate the mm_T kernel results
        restored_proj_input = self._decode_sbvr_rtn(proj_input, out_bvr, scales)
        out_emulated = self.weight_sbvr.forward(restored_proj_input)
        
        print_errors(out_emulated, out)
        
if __name__ == "__main__":
    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    W_BITS = 4
    device = "cuda:0"
    
    def randn_test():
        rtn_worker = PseudoRTN(MODEL_NAME, W_BITS, device, rtn_bits=7)
        x = torch.randn(1, 128, dtype=torch.float16, device="cuda:0")
        x_q, selected_indices = rtn_worker._apply_pseudo_rtn(x)
        print_errors(x, x_q)
        
    def proj_test(layer_idx, module_name):
        rtn_worker = PseudoRTN(MODEL_NAME, W_BITS, device, rtn_bits=7)
        rtn_worker.test_rtn_acc_module(layer_idx=layer_idx, module_name=module_name)
        
    def all_layers_test():
        rtn_worker = PseudoRTN(MODEL_NAME, W_BITS, device, rtn_bits=7)
        rtn_worker.test_rtn_acc_module_all_layers(n_layers=16)
        
    def sbvr_input_transform_test():
        rtn_worker = PseudoRTN(MODEL_NAME, W_BITS, device, rtn_bits=7)
        rtn_worker.test_sbvr_input_transform_rtn(layer_idx=0, module_name="q_proj")
        
    def rtn_sbvr_mm_T_test():
        rtn_worker = PseudoRTN(MODEL_NAME, W_BITS, device, rtn_bits=7)
        rtn_worker.test_rtn_sbvr_mm_T(layer_idx=0, module_name="q_proj")
        
        
    # randn_test()
    # proj_test(layer_idx=0, module_name="q_proj")
    # all_layers_test()
    # sbvr_input_transform_test()
    rtn_sbvr_mm_T_test()
    
    
   