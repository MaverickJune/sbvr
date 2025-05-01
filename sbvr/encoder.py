import torch
import itertools
import math
import numpy as np
from tqdm import tqdm
from .utils import g_str, y_str, b_str, r_str

class sbvr_encoder():
    def __init__(self, **kwargs):
        self.num_sums = kwargs.get("num_sums", 4)
        self.bvr_len = kwargs.get("bvr_len", 256)
        self.bvr_dtype = kwargs.get("bvr_dtype", torch.uint32)
        self.r_search_num = kwargs.get("r_search_num", 64)
        self.b_search_num = kwargs.get("b_search_num", 40)
        self.s_search_num = kwargs.get("s_search_num", 40)
        self.error_function = kwargs.get("error_function", "data_diff_mse")
        self.mse_window_size = kwargs.get("mse_window_size", 20)
        self.search_extend_ratio = kwargs.get("search_extend_ratio", 1.2)
        self.coeff_cache = kwargs.get("coeff_cache", None)
        self.cache_warmup_num = kwargs.get("cache_warmup_num", 10)
        self.acceptable_mse = kwargs.get("acceptable_mse", 10**-12)
        self.mse_history = kwargs.get("mse_history", [])
        self.search_batch_size = kwargs.get("search_batch_size", 0)
        self.group_idx = kwargs.get("group_idx", 0)
        self.cache_hits = kwargs.get("cache_hits", 0)
        self.num_coeff_cache_lines = kwargs.get("num_coeff_cache_lines", 0)
        self.extend_ratio = kwargs.get("extend_ratio", 1.2)
        self.verbose_level = kwargs.get("verbose_level", 0)
        self.compute_dtype = kwargs.get("compute_dtype", torch.float16)
        
    def _get_conf_str(self):
        conf_str = g_str("SBVR Encoder Config:") + \
            y_str("\n\tNumber of Summations: ") + str(self.num_sums) + \
            y_str("\n\tR search num: ") + str(self.r_search_num) + \
            y_str("\n\tB search num: ") + str(self.b_search_num) + \
            y_str("\n\tS search num: ") + str(self.s_search_num) + \
            y_str("\n\tMSE window size: ") + str(self.mse_window_size) + \
            y_str("\n\tSearch extend ratio: ") + \
            str(self.search_extend_ratio) + \
            y_str("\n\tCache warmup num: ") +str(self.cache_warmup_num) + \
            y_str("\n\tAcceptable MSE: ") + str(self.acceptable_mse) + \
            y_str("\n\tSearch batch size: ") + str(self.search_batch_size) + \
            y_str("\n\tExtend ratio: ") + str(self.extend_ratio)
        
        return conf_str
            
    def _get_result_str(self):
        result_str = y_str("\tCache hits: ") + str(self.cache_hits) + \
            y_str("\n\tNum coeff cache lines: ") + \
                str(self.num_coeff_cache_lines)
        return result_str
    
    def _check_coeff_cache_full(self):
        if self.num_coeff_cache_lines >= self.coeff_cache.shape[0]:
            return True
        return False
    
    def _get_bin_combs(self):
        if not hasattr(self, 'bin_combs'):
            self.bin_combs = torch.tensor(
                list(itertools.product([0, 1], repeat=self.num_sums)),
                dtype=self.compute_dtype, device=self.coeff_cache.device
            )
        return self.bin_combs
        
    def _get_additional_search_space(self, data, extended=False):
        search_budget = self.r_search_num * self.b_search_num * \
            self.s_search_num * 1.4
        if extended:
            search_budget *= self.search_extend_ratio**3
        search_num_per_dim = int(search_budget**(1/self.num_sums))
        
        data_max = torch.max(data)
        data_min = torch.min(data)
        
        dim_edges = torch.linspace(data_min, data_max, self.num_sums + 1,
                                    device=data.device, dtype=data.dtype)
        search_space = []
        for i in range (self.num_sums):
            search_range_i = \
                torch.linspace(dim_edges[i] - abs(dim_edges[i])*0.8, 
                               dim_edges[i+1] + abs(dim_edges[i])*0.8, 
                                search_num_per_dim + 1, device=data.device,
                                dtype=data.dtype)
            search_space.append(search_range_i)
        search_space = torch.cartesian_prod(*search_space)

        return search_space
    
    def _get_coeff_search_space_from_lists(self, r_list, b_list, s_list):
        exponents = torch.arange(self.num_sums, device=r_list.device) 
        search_space = r_list.unsqueeze(1) ** exponents.unsqueeze(0)
        all_vals = search_space @ self._get_bin_combs().T
        max = all_vals.max(dim=1)[0]
        min = all_vals.min(dim=1)[0]
        search_space = search_space / (max-min).unsqueeze(1)
        search_space = s_list.view(-1, 1, 1) * search_space.unsqueeze(0)
        search_space = b_list.view(-1, 1, 1, 1) + search_space.unsqueeze(0)
        search_space = search_space.view(-1, self.num_sums)
        
        return search_space
    
    def _get_coeff_search_space(self, data, extended=False):

        data_max = torch.max(data)
        data_avg = torch.mean(data)
        data_min = torch.min(data)
        data_95 = torch.quantile(data.to(torch.float), 0.95)

        r0_min = math.pi/6
        r0_max = 0.94
        r0_gran = (r0_max - r0_min) / (self.r_search_num / 2) 

        r1_max = math.pi*2/3
        r1_min = 1.06
        r1_gran = (r1_max - r1_min) / (self.r_search_num / 2) 
        b_max = abs(data_avg) * 2.0 / self.num_sums 
        if b_max < 0.3:
            b_max = 0.3
        b_min = -b_max
        b_gran = (b_max - b_min) / self.b_search_num 
        s_max = (data_max - data_min) * 1.1 
        s_min = 2 * data_95
        s_gran = (s_max - s_min) / self.s_search_num 
        
        if extended:
            if self.verbose_level > 1:
                print (r_str("\tUsing extended search space..."))
            r0_gran /= self.extend_ratio
            r1_gran /= self.extend_ratio
            b_gran /= self.extend_ratio
            s_gran /= self.extend_ratio
            
        if self.verbose_level > 2:
            print(b_str("\tNum_sums: ") + f"{self.num_sums}",
                    ", " + y_str("Data range: ") + 
                    f"{data_min:.4e} to {data_max:.4e}" +
                    ", " + y_str("avg: ") + f"{data_avg:.4e}")
            print(y_str("\t\tR0 search range: ") + 
                  f"{r0_min:.4e} to {r0_max:.4e}, " +
                y_str("search granularity: ") + f"{r0_gran:.4e}")
            print(y_str("\t\tR1 search range: ") + 
                  f"{r1_min:.4e} to {r1_max:.4e}, " +
                y_str("search granularity: ") + f"{r1_gran:.4e}")
            print(y_str("\t\tBias search range: ") + 
                f"{b_min:.4e} to {b_max:.4e}, " +
                y_str("search granularity: ") + f"{b_gran:.4e}")
            print(y_str("\t\tScale search range: ") + 
                f"{s_min:.4e} to {s_max:.4e}, " +
                y_str("search granularity: ") + f"{s_gran:.4e}")
        
        r0_list = -torch.arange(r0_min + r0_gran, r0_max + r0_gran, r0_gran, 
                              device=data.device, dtype=data.dtype)
        r1_list = -torch.arange(r1_min + r1_gran, r1_max + r1_gran, r1_gran, 
                              device=data.device, dtype=data.dtype)
        r_list = torch.cat((r0_list, r1_list))
        if s_gran != 0:
            s_list = torch.arange(s_min + s_gran, s_max + s_gran, s_gran, 
                                device=data.device, dtype=data.dtype)
        else:
            s_list = torch.tensor([s_min], device=data.device, dtype=data.dtype)
        if b_gran != 0:
            b_list = torch.arange(b_min, b_max, b_gran, 
                                  device=data.device, dtype=data.dtype)
        else:
            b_list = torch.tensor([b_min], device=data.device, dtype=data.dtype)
            
        search_space = \
            self._get_coeff_search_space_from_lists(r_list, b_list, s_list)
        org_search_space_len = search_space.shape[0]
        
        if self.num_sums <= 6:
            additional_search_space = \
                self._get_additional_search_space(data, extended)
            search_space = torch.cat((search_space, additional_search_space), 
                                     dim=0)
        
        _, indices = torch.sort(search_space.abs(), dim=1)
        search_space = torch.gather(search_space, dim=1, index=indices)
            
        return search_space, r_list, b_list, s_list, org_search_space_len
    
    def _data_diff_min_mse(self, data, candidate_matrix):
        n_ss_row = candidate_matrix.shape[0]
        n_ss_col = candidate_matrix.shape[1]
        
        data = data.view(1, -1, 1)
        candidate_matrix = candidate_matrix.view(n_ss_row, 1, n_ss_col) 
        
        diff = (data - candidate_matrix)**2

        diff_selected, coeff_comb_indices = diff.min(dim=-1) 
        mse = diff_selected.to(torch.float32).mean(dim=-1)
        
        min_idx = mse.argmin()
        coeff_comb_sel = coeff_comb_indices[min_idx]
        min_mse = mse[min_idx].item()
        
        return min_mse, min_idx, coeff_comb_sel
    
    def _get_min_mse_coeff(self, data, search_matrix):
        candidate_matrix = search_matrix @ self._get_bin_combs().T
        
        if self.error_function == "data_diff_mse":
            min_mse, min_idx, coeff_comb_sel = \
                self._data_diff_min_mse(data, candidate_matrix)

        return min_mse, min_idx, coeff_comb_sel
    
    def _search_coeff_bias_space(self, coeff_search_space, data, cutoff_mse):
        min_mse = float("inf")
        len_search_space = coeff_search_space.shape[0]
        best_coeff_idx = -1
        best_coeff_sel = -1
        # Loop over the bias values
        for search_start in range(0, len_search_space, self.search_batch_size):
            torch.cuda.empty_cache()
            search_end = \
                min(search_start + self.search_batch_size, len_search_space)
            coeff_list = coeff_search_space[search_start:search_end]
            # Call a method to get the index and MSE among these coefficients
            mse, min_idx, coeff_comb_sel = \
                self._get_min_mse_coeff(data, coeff_list)
            search_space_idx = search_start + min_idx
            if mse < min_mse:
                min_mse = mse
                best_coeff_idx = search_space_idx
                best_coeff_sel = coeff_comb_sel
                if min_mse < cutoff_mse:
                    break
        return min_mse, best_coeff_idx, best_coeff_sel
    
    def encode_data(self, data):
        min_mse = float("inf")
        self.group_idx += 1
        do_warmup = self.num_coeff_cache_lines < self.cache_warmup_num
        # Check cached search space
        if not do_warmup:
            # Setup the search space
            coeff_search_space = \
                self.coeff_cache[:self.num_coeff_cache_lines]
            # Setup the cutoff MSE 
            window_size = min(len(self.mse_history), 
                              self.mse_window_size)
            mse_window = self.mse_history[-window_size:]
            cutoff_mse = (sum(mse_window) / len(mse_window))*0.99
            if cutoff_mse < self.acceptable_mse:
                cutoff_mse = self.acceptable_mse

            # Search the cache for the best coeff and bias
            min_mse, best_coeff_idx, best_coeff_sel = \
                self._search_coeff_bias_space(coeff_search_space, 
                                              data, cutoff_mse) 
            if min_mse < cutoff_mse:
                self.cache_hits += 1
                return best_coeff_idx, best_coeff_sel
            else:
                if self.verbose_level > 1:
                    best_coeff_str = ['%.4f' % elem for elem in 
                              coeff_search_space[best_coeff_idx].tolist()]
                    hitrate = self.cache_hits / self.group_idx
                    print (b_str("\n\tGroup ") + f"{self.group_idx}: " 
                        + r_str("Cache Miss ") +
                        f"(Hitrate: {hitrate:.2f}) - " +
                        y_str("Coeff cache: ") +
                        f"{self.num_coeff_cache_lines}/" +
                        f"{self.coeff_cache.shape[0]}" +
                        y_str("\n\t\tCutoff MSE: ") + f"{cutoff_mse:.4e}" +
                        ", " + y_str("Best MSE: ") + f"{min_mse:.4e}" +
                        y_str("\n\t\tCoeff: ") + str(best_coeff_str))
        else:
            if self.verbose_level > 1:
                print(b_str("\n\tRun ") + f"{self.group_idx}: " +
                    r_str("Warming up cache... "))

        if not self._check_coeff_cache_full():
            hitrate = self.cache_hits / self.group_idx
            coeff_search_space, r_list, b_list, s_list, org_search_space_len = \
                self._get_coeff_search_space(data, hitrate > 0.6 or do_warmup)
            
            # Search the cache for the best coeff and bias  
            new_mse, new_coeff_idx, new_coeff_sel = \
                    self._search_coeff_bias_space(coeff_search_space, data, 
                                                  self.acceptable_mse)
                    
            if new_coeff_idx < org_search_space_len:
                new_b = b_list[new_coeff_idx // (len(s_list) * len(r_list))]
                new_s = s_list[new_coeff_idx // len(r_list) % len(s_list)]
                new_r = r_list[new_coeff_idx % len(r_list)]
            else:
                new_b = -1
                new_s = -1
                new_r = -1
                    
            if self.verbose_level > 1:
                new_coeff_str = ['%.4f' % elem for elem in 
                             coeff_search_space[new_coeff_idx].tolist()]
                print(g_str("\tNew MSE: ") + f"{new_mse:.4e}" +
                    ", " + y_str("(r, b, s): ") +
                    f"{new_r:.4e}, {new_b:.4e}, {new_s:.4e}" +
                    y_str("\n\t\tCoeff: ") + str(new_coeff_str))
            if new_mse >= min_mse:
                # If the new search space is NOT better than the cached one:
                if self.verbose_level > 1:
                    print(r_str("\t\tNo better coeff found: ") +
                          f"{new_mse:.4e} >= {min_mse:.4e}")
            else:
                # If the new search space is better than the cached one:
                # Cache the results
                coeff_diff = self.coeff_cache - \
                    coeff_search_space[new_coeff_idx].unsqueeze(0)
                avg_abs_coeff = coeff_search_space[new_coeff_idx].abs().sum(-1)
                mask = coeff_diff.abs().sum(-1) < avg_abs_coeff*0.0001
                if mask.any():
                    # If the coeff is already in the cache, use it
                    nonzero_idx = mask.nonzero(as_tuple=True)[0]
                    best_coeff_idx = nonzero_idx[0]
                else:
                    self.coeff_cache[self.num_coeff_cache_lines] =\
                        coeff_search_space[new_coeff_idx]
                    best_coeff_idx = self.num_coeff_cache_lines
                    self.num_coeff_cache_lines += 1

                # If caching was successful, update the output
                best_coeff_sel = new_coeff_sel
                self.mse_history.append(new_mse)
 
        return best_coeff_idx, best_coeff_sel
