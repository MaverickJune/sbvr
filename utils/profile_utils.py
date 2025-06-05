import torch
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from utils.utils import cleanup_memory, set_seed
import sbvr
from sbvr.utils import print_errors, get_errors, r_str, y_str, b_str, g_str
import torch.multiprocessing as mp
from utils import hadamard_utils
from tqdm import tqdm

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
        
        coeff_set_info = {
            "num_sums": num_sums,
            "bvr_len": bvr_len,
            "device": device,
            "coeff_set": torch.stack(coeff_set, dim=0).to(quantizer.compute_dtype).to(device)
        }
        
        if save_coeff_set:
            coeff_set_save_path = os.path.join(self.save_path, f"input_coeff_set.pt")
            torch.save(coeff_set_info, coeff_set_save_path)
        
        return coeff_set_info
    
    def get_per_state_encoding(self, coeff_set_size: int = 4, 
                                     num_sums: int = 8,
                                     bvr_len: int = 128,
                                     n_samples: int = 64,
                                     save_coeff_set: bool = False,
                                     down_proj_only: bool = False):
        if save_coeff_set:
            coeff_set_save_path = os.path.join(self.save_path, f"per_state_encoding")
            os.makedirs(coeff_set_save_path, exist_ok=True)
            
        mp.set_start_method('spawn', force=True)
        n_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {n_gpus}")
        
        if n_gpus == 0:
            raise ValueError("No GPUs found")
        
        curr_device = 0
        proc_list = [None for _ in range(n_gpus)]
        
        for layer_idx, layer_path in enumerate(self.input_file_paths):
            if proc_list[curr_device] is not None:
                proc_list[curr_device].join()
            if curr_device + 1 < n_gpus and proc_list[curr_device + 1] is not None:
                proc_list[curr_device + 1].join()
                
            proc_list[curr_device] = mp.Process(
                target=self.process_single_layer_input_encoding,
                args=(layer_idx, layer_path, curr_device, coeff_set_size, num_sums, bvr_len, n_samples, save_coeff_set, coeff_set_save_path,
                      down_proj_only)
            )
            proc_list[curr_device].start()
            curr_device = (curr_device + 1) % n_gpus
            
        for p in proc_list:
            p.join()
            
        print(b_str("per state input encoding complete\n"))

    @torch.inference_mode()
    def process_single_layer_input_encoding(self, layer_idx, layer_path, curr_device, coeff_set_size, 
                                            num_sums, bvr_len, n_samples, save_coeff_set, coeff_set_save_path, down_proj_only):
        input_info = torch.load(layer_path)
        target_types = ["down_proj"] if down_proj_only else ["k_proj", "o_proj", "gate_proj", "down_proj", "v_proj"]
        for type in target_types:
            print("getting per-state encoding for " + r_str(f"{layer_idx} {type}..."))
            
            coeff_set = []
            io_type = "input" if type in ["k_proj", "o_proj", "gate_proj", "down_proj"] else "output"
            input_info[io_type][type] = input_info[io_type][type].reshape(-1, bvr_len)
            print(b_str("finished loading target data, starting encoding..."))
            
            for _ in range(coeff_set_size):
                sample_indices = torch.randint(0, input_info[io_type][type].shape[0], (n_samples,))
                target_input = input_info[io_type][type][sample_indices].to(curr_device)
                quantizer = sbvr.sbvr(
                    encoder_config={
                        "num_sums": num_sums,
                        "bvr_len": bvr_len
                    },
                    verbose_level=1
                )
                coeff = quantizer._batched_input_encode(target_input)
                coeff_set.append(coeff)
            cleanup_memory()
            
            coeff_set_info = {
                "num_sums": num_sums,
                "bvr_len": bvr_len,
                "device": curr_device,
                "coeff_set": torch.stack(coeff_set, dim=0).to(quantizer.compute_dtype).to(curr_device)
            }
            print(b_str("finished encoding, saving coeff set...\n"))
            
            if save_coeff_set:
                torch.save(coeff_set_info, os.path.join(coeff_set_save_path, f"{layer_idx}_{type}.pt"))
            cleanup_memory()
                
    def encode_input_from_coeff_set(self, input: torch.Tensor, 
                                    coeff_set_info: dict = {}, 
                                    coeff_set_path: str = None,
                                    enable_oneshot_encoding: bool = False):
        if coeff_set_info != {} and coeff_set_path != None:
            raise ValueError("coeff_set_info and coeff_set_path cannot both be provided")
        if coeff_set_info == {} and coeff_set_path is None:
            raise ValueError("coeff_set_info or coeff_set_path must be provided")
        if coeff_set_path is not None:
            coeff_set_info = torch.load(coeff_set_path)
        
        device = coeff_set_info["device"]
        num_sums = coeff_set_info["num_sums"]
        bvr_len = coeff_set_info["bvr_len"]
        coeff_set = coeff_set_info["coeff_set"]
        
        input = input.to(device)
        if not enable_oneshot_encoding:
            quantizer = sbvr.sbvr(
                    encoder_config={
                        "num_sums": num_sums,
                        "bvr_len": bvr_len
                    },
                    verbose_level=1
                )
            quantizer._batched_encode_from_given_coeff_set(input, coeff_set)
            decoded_tensor = quantizer.decode()
        else:
            # one-shot encoding
            print(f"using oneshot encoding...")
            oneshot_quantizer = sbvr.core.sbvr_input(coeff_set, bvr_len, num_sums)
            bvr, coeff_idx = oneshot_quantizer.oneshot_input_encode(input)
            decoded_tensor = oneshot_quantizer.decode()
            
        errors, mse, max_error, min_error, std_dev = get_errors(input, decoded_tensor)
        return errors, mse, max_error, min_error, std_dev
        
    def test_sbvr_to_inputs(self, n_samples_per_input_type: int = 256,
                            coeff_set_info={}, 
                            coeff_set_path=None,
                            save_profile_results=False,
                            enable_oneshot_encoding=False,
                            per_state_encoding=False):
        input_types = ["k_proj", "o_proj", "gate_proj", "down_proj", "v_proj"]
        profile_results = {}
        for layer_idx, layer_path in enumerate(self.input_file_paths):
            input_info = torch.load(layer_path)
            for type in input_types:
                print(f"testing layer {layer_idx} {type}...")
                io_type = "input" if type in ["k_proj", "o_proj", "gate_proj", "down_proj"] else "output"
                
                input_info[io_type][type] = input_info[io_type][type].reshape(-1, input_info[io_type][type].shape[-1])
                sample_indices = torch.randint(0, input_info[io_type][type].shape[0], (n_samples_per_input_type,))
                target_input = input_info[io_type][type][sample_indices]
                
                if per_state_encoding:
                    coeff_set_path = os.path.join(self.save_path, f"per_state_encoding", f"{layer_idx}_{type}.pt")
                
                error, mse, max_error, min_error, std_dev = self.encode_input_from_coeff_set(target_input, coeff_set_info, coeff_set_path, 
                                                                                             enable_oneshot_encoding=enable_oneshot_encoding)
                profile_results[f"{layer_idx}_{type}"] = {
                    "error": error,
                    "mse": mse,
                    "max_error": max_error,
                    "min_error": min_error,
                    "std_dev": std_dev
                }
                print(r_str("Errors:   ") + 
                    y_str("MSE:  ") + f"{mse:.4e}" + ", " +
                    y_str("ABS Mean: ") + f"{torch.mean(error.abs()):.4e}" + ", " +
                    y_str("Max: ") + f"{max_error:.4e}" + ", " +
                    y_str("Min: ") + f"{min_error:.4e}" + ", " +
                    y_str("Std. Dev.: ") + f"{std_dev:.4e}\n")
                
        if save_profile_results:
            if per_state_encoding:
                torch.save(profile_results, os.path.join(self.save_path, f"per_state_profile_results.pt"))
            else:
                torch.save(profile_results, os.path.join(self.save_path, f"profile_results.pt"))
            
        return profile_results
    
    def printout_profile_results(self, profile_results: dict = {}, profile_results_path: str = None):
        if profile_results_path is not None:
            profile_results = torch.load(profile_results_path)
        if profile_results_path is None and profile_results == {}:
            raise ValueError("profile_results or profile_results_path must be provided")
        for key in profile_results.keys():
            layer_idx, type = key.split("_", 1)
            print(f"layer {layer_idx} {type}")
            print(r_str("Errors:   ") + 
                y_str("MSE:  ") + f"{profile_results[key]['mse']:.4e}" + ", " +
                y_str("ABS Mean: ") + f"{torch.mean(profile_results[key]['error'].abs()):.4e}" + ", " +
                y_str("Max: ") + f"{profile_results[key]['max_error']:.4e}" + ", " +
                y_str("Min: ") + f"{profile_results[key]['min_error']:.4e}" + ", " +
                y_str("Std. Dev.: ") + f"{profile_results[key]['std_dev']:.4e}")
            
    def apply_had_rot_to_down_proj(self):
        '''
        apply R4 rotation to down_proj for correct sbvr encoding
        '''
        for layer_path in tqdm(self.input_file_paths, desc="applying R4 rotation to down_proj", ncols=80):
            input_info = torch.load(layer_path)
            down_proj = input_info["input"]["down_proj"].to("cpu")
            for i in range(down_proj.shape[0]):
                target_tensor = down_proj[i].to("cuda:0")
                had_K, K = hadamard_utils.get_hadK(target_tensor.shape[-1])
                target_tensor = hadamard_utils.matmul_hadU_cuda(target_tensor, had_K, K)
                input_info["input"]["down_proj"][i] = target_tensor.to("cpu")
            cleanup_memory()
            torch.save(input_info, layer_path)
    
if __name__ == "__main__":
    profiler = input_profiler("meta-llama/Llama-3.2-1B", 4, 16, 16)
    
    # input_dist = profiler.sample_inputs()
    # input_dist = torch.load(os.path.join(profiler.save_path, f"input_dist_sample.pt"))
    # profiler.draw_input_distribution(input_dist)
    # coeff_set = profiler.get_sbvr_coeff_set_for_input(coeff_set_size=4, num_sums=8, bvr_len=128, device="cuda:0", save_coeff_cache=True)
    # coeff_set_path = os.path.join(profiler.save_path, f"input_coeff_set_info.pt")
    # profile_results = profiler.test_sbvr_to_inputs(coeff_set_path=coeff_set_path, save_profile_results=True)
    # print(profile_results)
    # profile_results_path = os.path.join(profiler.save_path, f"profile_results.pt")
    # profiler.printout_profile_results(profile_results_path=profile_results_path)
    # profiler_results = profiler.test_sbvr_to_inputs(coeff_set_path=os.path.join(profiler.save_path, f"input_coeff_set_info.pt"), save_profile_results=False, 
    #                                                 enable_oneshot_encoding=True)
    # profiler.get_per_state_encoding(coeff_set_size=4, num_sums=8, bvr_len=128, n_samples=128, save_coeff_set=True)
    # profiler_results = profiler.test_sbvr_to_inputs(n_samples_per_input_type=1, save_profile_results=False, enable_oneshot_encoding=True, per_state_encoding=True)
    
    # profiler.apply_had_rot_to_down_proj()
    profiler.get_per_state_encoding(coeff_set_size=4, num_sums=8, bvr_len=128, n_samples=128, save_coeff_set=True, down_proj_only=True)
    profiler_results = profiler.test_sbvr_to_inputs(save_profile_results=True, enable_oneshot_encoding=True, per_state_encoding=True)