import torch
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from utils.utils import cleanup_memory, set_seed
import sbvr

class input_profiler:
    def __init__(self, model_name, w_bits, a_bits, kv_bits,
                 save_path=None):
        self.model_name = self.model_name_formatter(model_name, w_bits, a_bits, kv_bits)
        self.input_storage_path = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            ), f"input_profile/{self.model_name}/layer_io"
        )
        self.n_layers = len([f for f in os.listdir(self.input_storage_path) if f.endswith(".pt")])
        self.input_file_paths = [
            os.path.join(self.input_storage_path, f"{i:03d}.pt") for i in range(self.n_layers)
        ]
        if save_path is None:
            self.save_path = os.path.dirname(self.input_storage_path)
        else:
            self.save_path = save_path
        
    def model_name_formatter(self, model_name, w_bits, a_bits, kv_bits):
        model_name = model_name.replace("/", "_")
        return f"{model_name}_{w_bits}_{a_bits}_{kv_bits}"
    
    def sample_inputs(self, group_size=128, n_samples_per_layer=20, save_input_dist=False,
                      save_name=f"input_dist_sample.pt"):
        n_samples_per_item = n_samples_per_layer // 5 # (k_proj, o_proj, gate_proj, down_proj, v_proj)
        distribution_board = torch.zeros(self.n_layers, group_size * n_samples_per_layer)
        
        for layer_idx, layer_path in enumerate(self.input_file_paths):
            print(f"sampling layer {layer_idx}...")
            input_info = torch.load(layer_path)
            print(f"loaded layer {layer_idx}")
            
            input_info["input"]["k_proj"] = input_info["input"]["k_proj"].reshape(-1, group_size)
            input_info["input"]["o_proj"] = input_info["input"]["o_proj"].reshape(-1, group_size)
            input_info["input"]["gate_proj"] = input_info["input"]["gate_proj"].reshape(-1, group_size)
            input_info["input"]["down_proj"] = input_info["input"]["down_proj"].reshape(-1, group_size)
            input_info["output"]["v_proj"] = input_info["output"]["v_proj"].reshape(-1, group_size)
            
            # generate random "n_samples_per_item" indices for each item
            k_indices = torch.randint(0, input_info["input"]["k_proj"].shape[0], (n_samples_per_item,))
            o_indices = torch.randint(0, input_info["input"]["o_proj"].shape[0], (n_samples_per_item,))
            gate_indices = torch.randint(0, input_info["input"]["gate_proj"].shape[0], (n_samples_per_item,))
            down_indices = torch.randint(0, input_info["input"]["down_proj"].shape[0], (n_samples_per_item,))
            v_indices = torch.randint(0, input_info["output"]["v_proj"].shape[0], (n_samples_per_item,))    
            
            # sample the inputs and store them in the distribution board
            distribution_board[layer_idx, :] = torch.cat([
                input_info["input"]["k_proj"][k_indices],
                input_info["input"]["o_proj"][o_indices],
                input_info["input"]["gate_proj"][gate_indices],
                input_info["input"]["down_proj"][down_indices],
                input_info["output"]["v_proj"][v_indices]
            ], dim=0).reshape(-1)
            print(f"sampled layer {layer_idx} \n")
            cleanup_memory()
         
        if save_input_dist:
            torch.save(distribution_board, os.path.join(self.save_path, save_name))
        return distribution_board
    
    def draw_input_distribution(self, input_dist: torch.Tensor = None):
        if input_dist is None:
            raise ValueError("input_dist is not provided")
        
        input_dist = input_dist.flatten()
        input_dist = torch.sort(input_dist)[0]
        fig_save_path = os.path.join(self.save_path, f"input_dist_sample.png")
        
        plt.figure(figsize=(10, 6))
        plt.hist(input_dist, bins=100, density=True, color="skyblue", edgecolor="black")

        # normal PDF overlay
        mu  = float(input_dist.mean())
        std = float(input_dist.std())
        xs  = np.linspace(mu - 4*std, mu + 4*std, 400)
        # plt.plot(xs, norm.pdf(xs, mu, std), lw=2)   # reference curve

        plt.title("Input Distribution")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.savefig(fig_save_path)
        plt.close()
        
    def get_sbvr_coeff_set_for_input(self, coeff_set_size: int = 4, 
                                     num_sums: int = 8,
                                     bvr_len: int = 128,
                                     device: str = "cuda:0",
                                     save_coeff_set: bool = False,
                                     save_input_dist: bool = False,
                                     input_dist_samples: list = None):
        
        if input_dist_samples is not None:
            if len(input_dist_samples) != coeff_set_size:
                raise ValueError("input_dist_samples must be a list of length coeff_set_size")
            else:
                input_dist_samples = [torch.load(os.path.join(self.save_path, f)).to(device) for f in input_dist_samples]
        else:
            input_dist_samples = [self.sample_inputs(group_size=128, save_input_dist=save_input_dist, save_name=f"input_dist_sample_{i}.pt").to(device) for i in range(coeff_set_size)]
            
        coeff_set = []
        for i in range(coeff_set_size):
            print(f"selecting coeff set {i}...")
            quantizer = sbvr.sbvr(
                encoder_config={
                    "num_sums": num_sums,
                    "bvr_len": bvr_len
                },
                verbose_level=1
            )
            input_dist = input_dist_samples[i]
            coeff = quantizer._batched_input_encode(input_dist)
            coeff_set.append(coeff)
        
        if save_coeff_set:
            coeff_set_save_path = os.path.join(self.save_path, f"input_coeff_set.pt")
            torch.save(coeff_set, coeff_set_save_path)
        
        return coeff_set
    
if __name__ == "__main__":
    profiler = input_profiler("meta-llama/Llama-3.2-1B", 4, 16, 16)
    
    # input_dist = profiler.sample_inputs()
    # input_dist = torch.load(os.path.join(profiler.save_path, f"input_dist_sample.pt"))
    # profiler.draw_input_distribution(input_dist)
    # coeff_set = profiler.get_sbvr_coeff_set_for_input(coeff_set_size=4, num_sums=8, bvr_len=128, device="cuda:0", save_coeff_cache=True)
        
        