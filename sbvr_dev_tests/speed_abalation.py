import torch
from sbvr import sbvr
from sbvr import load as sbvr_load
from sbvr.core import sbvrizer, load_sbvrizer, input_sbvr_mm_T
from sbvr.utils import print_errors, get_errors, r_str, y_str, b_str, g_str
from contextlib import nullcontext

import os
import sys

class SpeedAbalationHelper:
    def __init__(self, model_name, w_bits, device="cuda:0"):
        self.model_name = self.model_name_formatter(model_name, w_bits)
        self.weight_sbvr_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            ), f"quantized_model/{self.model_name}"
        )
        self.input_sbvrizer_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            ), f"input_profile/{self.model_name}/per_state_encoding"
        )
        self.input_data_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            ), f"input_profile/{self.model_name}/layer_io"
        )
        self.device = device
        self.target_input = None
        self.weight_sbvr = None
        self.input_sbvrizer = None
        
    def model_name_formatter(self, model_name, w_bits, a_bits=16, kv_bits=16):
        model_name = model_name.replace("/", "_")
        return f"{model_name}_{w_bits}_{a_bits}_{kv_bits}"
    
    def convert_layer_to_input_filename(self, layer_idx):
        return f"{layer_idx:03d}.pt"
    
    def convert_params_to_input_sbvrizer_filename(self, layer_idx=-1, module_name=None):
        module_name_list = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
        module_name_to_sbvrizer_dict = {
            "q_proj": "k_proj",
            "k_proj": "k_proj",
            "v_proj": "k_proj",
            "o_proj": "o_proj",
            "up_proj": "gate_proj",
            "gate_proj": "gate_proj",
            "down_proj": "down_proj"
        }
        if layer_idx == -1 or module_name == None:
            raise ValueError("layer_idx and module_name must be provided")
        if module_name not in module_name_list:
            raise ValueError(f"module_name must be one of {module_name_list}")
        
        return f"{layer_idx}_{module_name_to_sbvrizer_dict[module_name]}.pt"
    
    def convert_params_to_weight_sbvr_filename(self, layer_idx=-1, module_name=None):
        module_name_list = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]
        if layer_idx == -1 or module_name == None:
            raise ValueError("layer_idx and module_name must be provided")
        if module_name not in module_name_list:
            raise ValueError(f"module_name must be one of {module_name_list}")
        
        module_indicator = "self_attn" if module_name in ["q_proj", "k_proj", "v_proj", "o_proj"] else "mlp"
        
        return f"sbvr_layer_{layer_idx}_{module_indicator}.{module_name}.module.pt"
    
    @torch.inference_mode()
    def prepare_speed_abalation(self, layer_idx=-1, module_name=None):
        # prepare the input data
        module_name_to_sbvrizer_dict = {
            "q_proj": "k_proj",
            "k_proj": "k_proj",
            "v_proj": "k_proj",
            "o_proj": "o_proj",
            "up_proj": "gate_proj",
            "gate_proj": "gate_proj",
            "down_proj": "down_proj"
        }
        input_path = os.path.join(self.input_data_path, self.convert_layer_to_input_filename(layer_idx))
        io_indicator = "output" if module_name in ["v_proj"] else "input"
        batched_qkv_proj_input = torch.load(input_path, map_location=self.device)[io_indicator][module_name_to_sbvrizer_dict[module_name]]
        batched_qkv_proj_input = batched_qkv_proj_input.reshape(-1, batched_qkv_proj_input.shape[-1])
        select_idx = torch.randint(0, batched_qkv_proj_input.shape[0], (1,))
        self.target_input = batched_qkv_proj_input[select_idx]
        
        # prepare the weight sbvr
        weight_sbvr_path = os.path.join(self.weight_sbvr_path, self.convert_params_to_weight_sbvr_filename(layer_idx, module_name))
        self.weight_sbvr = sbvr_load(weight_sbvr_path, device=self.device)
        
        # prepare the input sbvrizer
        input_sbvrizer_path = os.path.join(self.input_sbvrizer_path, self.convert_params_to_input_sbvrizer_filename(layer_idx, module_name))
        self.input_sbvrizer = load_sbvrizer(input_sbvrizer_path, device=self.device).to(torch.float16)
    
    @torch.inference_mode()
    def test_speed_abalation(
        self,
        warmup_iters: int = 10,
        measure_iters: int = 100,
        sync_ctx=nullcontext(),
    ):
        if any(
            v is None
            for v in (self.target_input, self.weight_sbvr, self.input_sbvrizer)
        ):
            raise RuntimeError(
                "Run prepare_speed_abalation(...) before calling this method."
            )
        
        def _record(fn):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            out = fn()
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end), out
        
        def _mean_std(times):
            t = torch.tensor(times, dtype=torch.float32, device="cpu")
            return float(t.mean()), float(t.std(unbiased=False))
        
        # warmup
        with torch.no_grad(), sync_ctx:
            for _ in range(warmup_iters):
                hs = self.input_sbvrizer(self.target_input, mode=1)
                _ = input_sbvr_mm_T(
                    self.input_sbvrizer.bvr, self.input_sbvrizer.coeff_idx, self.input_sbvrizer.coeff_set, self.weight_sbvr
                )
        torch.cuda.synchronize()
        
        # measure: input_sbvrization
        input_times = []
        with torch.no_grad(), sync_ctx:
            for _ in range(measure_iters):
                t, _ = _record(
                    lambda: self.input_sbvrizer(self.target_input, mode=1)
                )
                input_times.append(t)
                
        # measure: weight gemm
        weight_times = []
        with torch.no_grad(), sync_ctx:
            for _ in range(measure_iters):
                t, _ = _record(
                    lambda: input_sbvr_mm_T(
                        self.input_sbvrizer.bvr, self.input_sbvrizer.coeff_idx, self.input_sbvrizer.coeff_set, self.weight_sbvr
                    )
                )
                weight_times.append(t)
                
        # return the results  
        in_mean, in_std = _mean_std(input_times)
        w_mean, w_std = _mean_std(weight_times)
        
        self.results = {
            "input_sbvrize_ms_mean": in_mean,
            "input_sbvrize_ms_std": in_std,
            "weight_mm_ms_mean": w_mean,
            "weight_mm_ms_std": w_std,
        }
        
        return self.results
    
    def print_results(self):
        results_log = r_str("\nSpeed Abalation Results:") + \
            g_str(f"\n\tInput SBVR: ") + f"{self.results['input_sbvrize_ms_mean']:.4f} ± {self.results['input_sbvrize_ms_std']:.4f} ms" + \
            g_str(f"\n\tWeight MM: ") + f"{self.results['weight_mm_ms_mean']:.4f} ± {self.results['weight_mm_ms_std']:.4f} ms\n"
            
        print(results_log)
        

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-1B"
    w_bits = 4
    device = "cuda:0"
    
    helper = SpeedAbalationHelper(model_name, w_bits, device)
    helper.prepare_speed_abalation(layer_idx=0, module_name="down_proj")
    results = helper.test_speed_abalation()
    helper.print_results()
    
    
    