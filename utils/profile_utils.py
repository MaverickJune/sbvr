import torch
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from utils.utils import cleanup_memory

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
    
    def sample_inputs(self, group_size=128, n_samples_per_layer=20, save_input_dist=False):
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
            torch.save(distribution_board, os.path.join(self.save_path, f"input_dist_sample.pt"))
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
        plt.plot(xs, norm.pdf(xs, mu, std), lw=2)   # reference curve

        plt.title("Input Distribution vs. Best-Fit Normal")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.savefig(fig_save_path)
        plt.close()
    
if __name__ == "__main__":
    profiler = input_profiler("meta-llama/Llama-3.2-1B", 4, 16, 16)
    input_dist = profiler.sample_inputs()
    profiler.draw_input_distribution(input_dist)
            
        
        