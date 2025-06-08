import torch
import itertools
import math
import numpy as np
from tqdm import tqdm
from .encoder import sbvr_encoder
from .utils import g_str, y_str, b_str, r_str, sbvr_serialized, cleanup_memory
from sbvr.sbvr_cuda import _sbvr_mm_T
import os
import sys
        
class sbvr(torch.nn.Module):
    def __init__(self, 
                 data: torch.Tensor = None, 
                 encoder_config: dict = None,
                 device: torch.device = None,
                 serialized: sbvr_serialized = None,
                 verbose_level: int = 1):
        super(sbvr, self).__init__()
        _device = device if device is not None else \
            torch.device("cuda") if torch.cuda.is_available() \
                else torch.device("cpu")
        self.verbose_level = verbose_level
        
        if data is not None and serialized is not None:
            raise ValueError(
                r_str("Cannot provide both data and serialized SBVR"))
    
        # Load serialized SBVR object
        if serialized is not None:
            if not isinstance(serialized, sbvr_serialized):
                raise ValueError(
                    r_str("Serialized SBVR object is not valid"))
            self.num_sums, self.bvr_len, self.compute_dtype, \
                self.bvr_dtype, self.original_dtype, self.original_data_shape, \
                bvr, coeff_idx, coeff_cache, self.input_num_sums, \
                                input_coeff = serialized.deserialize_sbvr()
                
            self.bvr = torch.nn.Parameter(bvr.to(_device), requires_grad=False)
            self.coeff_idx = torch.nn.Parameter(coeff_idx.to(_device), 
                                                requires_grad=False)
            self.coeff_cache = torch.nn.Parameter(coeff_cache.to(_device),
                                                  requires_grad=False)
            if input_coeff is not None:
                self.input_coeff = torch.nn.Parameter(
                    input_coeff.to(_device), requires_grad=False)
            else:
                self.input_coeff = None
        else:
            if encoder_config is None:
                encoder_config = {} 
            self.encoder = sbvr_encoder(**encoder_config)
            self.encoder.verbose_level = verbose_level
            self.num_sums = self.encoder.num_sums
            self.bvr_len = self.encoder.bvr_len
            self.compute_dtype = self.encoder.compute_dtype
            self.bvr_dtype = self.encoder.bvr_dtype
            
            self.enable_blockwise_gptq = self.encoder.enable_blockwise_gptq
            
            if self.num_sums > 11 and self.compute_dtype == torch.float16:
                raise UserWarning(
                    r_str("Warning: compute_dtype float16 does not have "
                           "sufficient precision for num_sums > 11."))
            if self.bvr_len % self._get_bvr_num_bits() != 0:
                raise ValueError(
                    r_str("BVR length must be a multiple of ") +
                        f"{self._get_bvr_num_bits()}")
                
            self.input_num_sums = -1
        
            self.bvr = None
            self.coeff_idx = None
            self.coeff_cache = None
            self.input_coeff = None
            if data is not None:
                self._batched_encode(data.to(_device).to(self.compute_dtype))
    
    def _get_bvr_num_bits(self):
        if not hasattr(self, 'bvr_num_bits'):
            self.bvr_num_bits = \
                torch.tensor(0, dtype=self.bvr_dtype).element_size() * 8
        return self.bvr_num_bits
    
    def _get_padded_data_shape(self):
        if not hasattr(self, 'padded_data_shape'):
            if self.original_data_shape is not None:
                self.padded_data_shape = list(self.original_data_shape)
                self.padded_data_shape[-1] = \
                    (self.original_data_shape[-1] + self.bvr_len - 1) // \
                    self.bvr_len * self.bvr_len
                self.padded_data_shape = \
                    torch.Size(list(self.padded_data_shape))
            else:
                self.padded_data_shape = None
        return self.padded_data_shape
    
    def _get_padded_input_shape(self, input):
        padded_input_shape = list(input.shape)
        padded_input_shape[-1] = \
            (input.shape[-1] + self.bvr_len - 1) // self.bvr_len * self.bvr_len
        padded_input_shape = torch.Size(list(padded_input_shape))
        return padded_input_shape

    def _get_dummy_bias(self):
        if not hasattr(self, 'dummy_bias'):
            self.dummy_bias = torch.zeros([0],
                                          dtype=self.compute_dtype,
                                          device=self.coeff_cache.device)
        return self.dummy_bias
        
    def _get_all_points(self, coeff: torch.tensor):
        if not hasattr(self, 'bin_combs'):
            self.bin_combs = torch.tensor(
                list(itertools.product([0, 1], repeat=self.num_sums)),
                dtype=self.compute_dtype, device=self.coeff_cache.device
            )
        return coeff @ self.bin_combs.T
    
    def prepare_encoding(self, data):
        print (r_str("Preparing SBVR encoding: ") + str(data.shape))
        self.original_dtype = data.dtype
        self.original_data_shape = data.shape
                
        if data.device.type == 'cuda':
            elem_size = torch.tensor(0, dtype=self.compute_dtype).element_size()
            diff_mat_size = 3 * self.encoder.extend_ratio * (2**self.num_sums) \
                                * self.bvr_len * elem_size
            total_mem = torch.cuda.mem_get_info(data.device)[0]
            self.encoder.search_batch_size = \
                int(total_mem * 0.8 / diff_mat_size)
        else:
            self.encoder.search_batch_size = 1024

        # Pad the data to the nearest multiple of bvr_len
        if self.original_data_shape != self._get_padded_data_shape():
            # raise ValueError(
            #     r_str("Data shape must be a multiple of bvr_len"))
            data_padded = torch.zeros(self._get_padded_data_shape(), 
                                      dtype=self.compute_dtype, 
                                      device=data.device)
            slices = tuple(slice(0, s) for s in data.shape)
            data_padded[slices] = data
        else:
            data_padded = data
        
        self.encoder.coeff_cache = torch.zeros((2**16, self.num_sums), 
                        dtype=self.compute_dtype, device=data.device)
        self.coeff_cache = self.encoder.coeff_cache
        self.coeff_sel = torch.empty((data_padded.numel()), 
                                dtype=torch.int32,
                                device=data.device)
        self.coeff_idx = torch.empty((data_padded.numel() // self.bvr_len),
                                dtype=torch.uint16, 
                                device=data.device)
        
        return data_padded
    
    @torch.inference_mode()
    def _batched_input_encode(self, data):
        # First, apply sbvr encoding to the data to fill the coeff cache
        data_padded = self.prepare_encoding(data)
        
        if self.verbose_level > 0:
            print(self.encoder._get_conf_str())
        
        if self.verbose_level > -1:
            group_iter = tqdm(range(self.coeff_idx.shape[0]), ncols=80, 
                      desc=b_str("Encoding SBVR groups"), unit="g")
        else:
            group_iter = range(self.coeff_idx.shape[0])
        
        for i in group_iter:
            group_data = data_padded.flatten()[i * self.bvr_len: 
                                                (i + 1) * self.bvr_len]
            self.iterative_encoding(group_data, i)
            
        # Let the coeff set in the self.cache be the candidates
        print(r_str("Selecting the best coeff set from the cache ..."))
        
        coeff_cache_mse = torch.zeros(self.encoder.num_coeff_cache_lines, device=data.device, dtype=data.dtype)
        for i in group_iter:
            group_data = data_padded.flatten()[i * self.bvr_len: (i + 1) * self.bvr_len]
            coeff_cache_mse_partial = self.encoder.get_input_data_mse_from_cache(group_data)
            coeff_cache_mse += coeff_cache_mse_partial
        coeff_cache_mse = coeff_cache_mse / self.encoder.group_idx
        min_idx = coeff_cache_mse.argmin(dim=-1)
        best_mse = coeff_cache_mse[min_idx]
        best_coeff = self.encoder.coeff_cache[min_idx]
        
        print(b_str("Best MSE: ") + f"{best_mse:.4e}")
        
        return best_coeff
    
    @torch.inference_mode()
    def _batched_encode_from_given_coeff_set(self, data, coeff_set):
        if not isinstance(coeff_set, torch.Tensor):
            raise ValueError(r_str("coeff_set must be a torch.Tensor"))
        if coeff_set.shape[-1] != self.num_sums:
            raise ValueError(r_str("coeff_set must have the same number of columns as the num_sums"))
        if coeff_set.dtype != self.compute_dtype:
            raise ValueError(r_str("coeff_set must have the same dtype as the compute_dtype"))
        
        data_padded = self.prepare_encoding(data)
        if self.verbose_level > -1:
            group_iter = tqdm(range(self.coeff_idx.shape[0]), ncols=80, 
                      desc=b_str("Encoding SBVR groups"), unit="g")
        else:
            group_iter = range(self.coeff_idx.shape[0])
        
        for i in group_iter:
            group_data = data_padded.flatten()[i * self.bvr_len: 
                                                (i + 1) * self.bvr_len]
            self.iterative_encoding_from_given_coeff_set(group_data, coeff_set, i)
        
        # forcefully update cache info to integrate the code
        self.encoder.coeff_cache = coeff_set
        self.encoder.num_coeff_cache_lines = coeff_set.shape[0]
        
        self.finalize_encoding(input_sbvr_mode=True)
            
    @torch.inference_mode()
    def _batched_encode(self, data):
        data_padded = self.prepare_encoding(data)
        
        if self.verbose_level > 0:
            print(self.encoder._get_conf_str())
        
        if self.verbose_level > -1:
            group_iter = tqdm(range(self.coeff_idx.shape[0]), ncols=80, 
                      desc=b_str("Encoding SBVR groups"), unit="g")
        else:
            group_iter = range(self.coeff_idx.shape[0])
        
        for i in group_iter:
            group_data = data_padded.flatten()[i * self.bvr_len: 
                                                (i + 1) * self.bvr_len]
            self.iterative_encoding(group_data, i)
    
        self.finalize_encoding()
        
    def iterative_encoding(self, data, idx):
        torch.cuda.empty_cache()
        coeff_idx, coeff_sel = \
                self.encoder.encode_data(data.to(self.compute_dtype))
        self.coeff_idx[idx] = coeff_idx
        self.coeff_sel[idx * self.bvr_len:(idx + 1) * self.bvr_len] = coeff_sel
        all_points = self._get_all_points(self.encoder.coeff_cache[coeff_idx])
        decoded_data = all_points[coeff_sel]
        # print(f"Decoded data: {decoded_data}")
        return decoded_data
    
    def iterative_encoding_from_given_coeff_set(self, data, coeff_set, idx):
        coeff_idx, coeff_sel = \
                self.encoder.encode_data_from_given_coeff_set(data.to(self.compute_dtype), coeff_set)
        self.coeff_idx[idx] = coeff_idx
        self.coeff_sel[idx * self.bvr_len:(idx + 1) * self.bvr_len] = coeff_sel
        all_points = self._get_all_points(coeff_set[coeff_idx])
        decoded_data = all_points[coeff_sel]
        return decoded_data
        
    def finalize_encoding(self, input_sbvr_mode=False):
        if self.enable_blockwise_gptq:
            # reshape self.coeff_sel and coeff_idx
            self.coeff_idx = self.coeff_idx.reshape(-1, self.padded_data_shape[0]).transpose(0, 1).contiguous().flatten()
            target_sel_shape = (self.padded_data_shape[-1] // self.bvr_len, self.padded_data_shape[0], self.bvr_len)
            self.coeff_sel = self.coeff_sel.reshape(target_sel_shape).permute(1, 0, 2).contiguous().flatten()
        
        bvr = self._change_coeff_sel_to_bvr()
        bvr = bvr.view(self.num_sums, -1, self._get_padded_data_shape()[-1] // \
                                                self._get_bvr_num_bits())
        bvr = bvr.permute(2, 1, 0).contiguous()
        self.bvr = torch.nn.Parameter(bvr, requires_grad=False)
        
        coeff_cache = \
            self.encoder.coeff_cache[:self.encoder.num_coeff_cache_lines]
        coeff_cache = coeff_cache.contiguous()
        if self.encoder.num_coeff_cache_lines <= 256:
            self.coeff_idx = self.coeff_idx.to(torch.uint8)
        coeff_idx = self.coeff_idx.view(-1, self.bvr.shape[0] // \
                                (self.bvr_len // self._get_bvr_num_bits()))
        coeff_idx = coeff_idx.transpose(0, 1).contiguous()
            
        self.coeff_idx = torch.nn.Parameter(coeff_idx, requires_grad=False)
        self.coeff_cache = torch.nn.Parameter(coeff_cache, requires_grad=False)
        
        if self.verbose_level > 0:
            if not input_sbvr_mode:
                print(b_str("Encoding complete."))
                print(self.encoder._get_result_str())
                print(self.get_sbvr_info())       
        del self.encoder
            
    def _dec2bin(self, x, bits):
        # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

    def _bin2dec(self, b, bits):
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
        return torch.sum(mask * b, -1)
            
    def _change_coeff_sel_to_bvr(self):
        coeff_sel_len = self._get_padded_data_shape().numel()
        num_bits = self._get_bvr_num_bits()
        bvr = torch.zeros((self.num_sums, (coeff_sel_len // num_bits)),
                dtype=self.bvr_dtype, device=self.coeff_cache.device)
        powers = 2 ** torch.arange(num_bits, dtype=torch.int64, 
                                   device=self.coeff_cache.device)
        iter_size = 65536
        for i in range(0, coeff_sel_len, iter_size):
            max_i = min(i + iter_size, coeff_sel_len)
            coeff_sel_i = self.coeff_sel[i:max_i]
            bin_vec = self._dec2bin(coeff_sel_i, self.num_sums).to(torch.int64)
            bin_vec = \
                bin_vec.transpose(0, 1).reshape(self.num_sums, -1, num_bits)
            bvr_i = torch.sum(bin_vec * powers.unsqueeze(0), dim=2)
            bvr[:, i//32:max_i//32] = bvr_i
        del self.coeff_sel
        return bvr
     
    def _change_bvr_to_coeff_sel(self):
        coeff_sel_len = self._get_padded_data_shape().numel()
        bvr = self.bvr.permute(2, 1, 0).contiguous().view(self.num_sums, -1)
        bvr = bvr.view(self.num_sums, -1)
        num_bits = self._get_bvr_num_bits()
        powers = 2 ** torch.arange(num_bits, 
                                   dtype=torch.int64, 
                                   device=self.coeff_cache.device)
        coeff_sel = torch.empty((bvr.shape[1] * num_bits),
                               dtype=torch.int32, 
                               device=self.coeff_cache.device)
        iter_size = 2048
        for i in range(0, bvr.shape[1], iter_size):
            max_i = min(i + iter_size, bvr.shape[1])
            bvr_i = bvr[:, i:max_i].to(torch.int64)
            bin_vec = ((bvr_i.unsqueeze(-1) & powers) != 0).to(torch.int32)
            bin_vec = bin_vec.view(self.num_sums, -1)
            max_coeff_i = min(max_i*num_bits, coeff_sel.shape[0])
            bin_vec_trunc = bin_vec[:, :max_coeff_i].transpose(0, 1)
            coeff_sel_i = self._bin2dec(bin_vec_trunc, self.num_sums)
            coeff_sel[i*num_bits:max_coeff_i] = coeff_sel_i.view(-1)

        return coeff_sel[:coeff_sel_len]
    
    def _serialize(self):
        return sbvr_serialized(
            self.num_sums,
            self.bvr_len,
            self.compute_dtype,
            self.bvr_dtype,
            self.original_dtype,
            self.original_data_shape,
            self.bvr,
            self.coeff_idx,
            self.coeff_cache,
            self.input_num_sums,
            self.input_coeff
        )
    
    @torch.inference_mode()
    def save(self, filename):   
        if self.verbose_level > 0:
            print(b_str("Saving SBVR object to: ") + filename)
            print(self.get_sbvr_info()) 
        serialized_sbvr = self._serialize()
        torch.save(serialized_sbvr, filename)
            
    @torch.inference_mode()
    def decode(self):
        decoded_tensor = torch.empty(self._get_padded_data_shape(),
                                      dtype=self.original_dtype,
                                      device=self.coeff_cache.device)
        num_bvr = self.coeff_idx.numel()
        coeff_sel = self._change_bvr_to_coeff_sel()
        coeff_idx = self.coeff_idx.transpose(0, 1).contiguous().flatten()
        for i in range(num_bvr):
            group_start = i * self.bvr_len
            group_end = \
                min(group_start + self.bvr_len, decoded_tensor.numel())
            group_coeff = self.coeff_cache[coeff_idx[i].item()]
            group_coeff_sel = coeff_sel[group_start:group_end]
            group_all_points = self._get_all_points(group_coeff)
            group_data = group_all_points[group_coeff_sel]
            decoded_tensor.flatten()[group_start:group_end] = group_data
            
        # Truncate the tensor to the original shape
        if self.original_data_shape != self._get_padded_data_shape():
            slices = tuple(slice(0, s) for s in self.original_data_shape)
            decoded_tensor = decoded_tensor[slices]
        
        return decoded_tensor
    
    def get_sbvr_info(self):
        info_str = g_str("SBVR Info:") + \
        y_str("\n\tNumber of Summations: ") + str(self.num_sums) + \
        y_str("\n\tBVR Length: ") + str(self.bvr_len) + \
        y_str("\n\tCompute Data Type: ") + str(self.compute_dtype) + \
        y_str("\n\tBVR Data Type: ") + str(self.bvr_dtype) + \
        y_str("\n\tOriginal Data Type: ") + str(self.original_dtype) + \
        y_str("\n\tOriginal Data Shape: ") + str(self.original_data_shape) + \
        y_str("\n\tNumber of Coefficient Cache Lines: ") + \
        y_str("\n\tBVR Tensor Shape: ") + str(self.bvr.shape) + \
        y_str("\n\tCoefficient Index Shape: ") + str(self.coeff_idx.shape) + \
        y_str("\n\tCoefficient Cache Shape: ") + \
            str(self.coeff_cache.shape) + \
        y_str("\n\tInput Number of Summations: ") + \
            str(self.input_num_sums) + \
        y_str("\n\tInput Coefficient Shape: ") + \
            str(self.input_coeff.shape if self.input_coeff is not None 
                else "Input Coefficient not set")
            
        return info_str
    
def mm_T(lhs, rhs, bias):
    lhs_bvr = lhs.bvr
    lhs_coeff_idx = lhs.coeff_idx
    lhs_coeff_cache = lhs.coeff_cache
    rhs_bvr = rhs.bvr
    rhs_coeff_idx = rhs.coeff_idx
    rhs_coeff_cache = rhs.coeff_cache
    if bias is None:
        bias = lhs._get_dummy_bias()
    return _sbvr_mm_T(lhs_bvr, lhs_coeff_idx, lhs_coeff_cache,
                rhs_bvr, rhs_coeff_idx, rhs_coeff_cache, bias)
    
def load(filename, device=None, verbose_level=1) -> sbvr:
    serialized_sbvr = torch.load(filename)
    sbvr_obj = sbvr(serialized=serialized_sbvr, 
                    verbose_level=verbose_level, device=device)
    sbvr_obj.verbose_level = verbose_level
    if verbose_level > 0:
        print(b_str("Loaded SBVR object from: ") + filename)
        print(sbvr_obj.get_sbvr_info())
    return sbvr_obj


class sbvr_input:
    def __init__(self, coeff_set, bvr_len, num_sums):
        self.coeff_set = coeff_set
        self.bvr_len = bvr_len
        self.num_sums = num_sums
        self.bvr_dtype = torch.uint32
        self.conversion_complete = False
        
    def _get_bvr_num_bits(self):
        if not hasattr(self, 'bvr_num_bits'):
            self.bvr_num_bits = \
                torch.tensor(0, dtype=self.bvr_dtype).element_size() * 8
        return self.bvr_num_bits
    
    def _dec2bin(self, x, bits):
        # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()
    
    def _bin2dec(self, b, bits):
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
        return torch.sum(mask * b, -1)
    
    def _get_padded_data_numel(self):
        if not hasattr(self, 'padded_data_shape'):
            raise ValueError(r_str("padded_data_shape not set"))
        return self.padded_data_shape[0] * self.padded_data_shape[1]
    
    @torch.inference_mode()
    def _get_candidate_mm_ext(self, data):
        if not self.conversion_complete:
            self.coeff_set = self.coeff_set.to(data.device).to(data.dtype)
            self.conversion_complete = True
        dtype = data.dtype
        device = data.device
        if not hasattr(self, 'bin_combs'):
            self.bin_combs = torch.tensor(
                list(itertools.product([0, 1], repeat=self.num_sums)),
                dtype=dtype, device=device
            )
        if not hasattr(self, 'candidate_matrix'):
            self.candidate_matrix = self.coeff_set @ self.bin_combs.T
        self.bin_combs = self.bin_combs.to(device).to(dtype)
        self.candidate_matrix = self.candidate_matrix.to(device).to(dtype)
        
        return self.candidate_matrix
    
    @torch.inference_mode()
    def _get_all_points(self, coeff):
        return coeff @ self.bin_combs.T
    
    def _change_coeff_sel_to_bvr(self):
        coeff_sel_len = self._get_padded_data_numel()
        num_bits = self._get_bvr_num_bits()
        bvr = torch.zeros((self.num_sums, (coeff_sel_len // num_bits)),
                dtype=self.bvr_dtype, device=self.data_device)
        powers = 2 ** torch.arange(num_bits, dtype=torch.int64, 
                                   device=self.data_device)
        iter_size = 65536
        for i in range(0, coeff_sel_len, iter_size):
            max_i = min(i + iter_size, coeff_sel_len)
            coeff_sel_i = self.coeff_sel[i:max_i]
            bin_vec = self._dec2bin(coeff_sel_i, self.num_sums).to(torch.int64)
            bin_vec = \
                bin_vec.transpose(0, 1).reshape(self.num_sums, -1, num_bits)
            bvr_i = torch.sum(bin_vec * powers.unsqueeze(0), dim=2)
            bvr[:, i//32:max_i//32] = bvr_i
        del self.coeff_sel
        return bvr
    
    def _change_bvr_to_coeff_sel(self):
        coeff_sel_len = self._get_padded_data_numel()
        bvr = self.bvr.permute(2, 1, 0).contiguous().view(self.num_sums, -1)
        bvr = bvr.view(self.num_sums, -1)
        num_bits = self._get_bvr_num_bits()
        powers = 2 ** torch.arange(num_bits, 
                                   dtype=torch.int64, 
                                   device=self.data_device)
        coeff_sel = torch.empty((bvr.shape[1] * num_bits),
                               dtype=torch.int32, 
                               device=self.data_device)
        iter_size = 2048
        for i in range(0, bvr.shape[1], iter_size):
            max_i = min(i + iter_size, bvr.shape[1])
            bvr_i = bvr[:, i:max_i].to(torch.int64)
            bin_vec = ((bvr_i.unsqueeze(-1) & powers) != 0).to(torch.int32)
            bin_vec = bin_vec.view(self.num_sums, -1)
            max_coeff_i = min(max_i*num_bits, coeff_sel.shape[0])
            bin_vec_trunc = bin_vec[:, :max_coeff_i].transpose(0, 1)
            coeff_sel_i = self._bin2dec(bin_vec_trunc, self.num_sums)
            coeff_sel[i*num_bits:max_coeff_i] = coeff_sel_i.view(-1)

        return coeff_sel[:coeff_sel_len]
        
    @torch.inference_mode()
    def oneshot_input_encode(self, data, use_oom_avoidance=False):
        """
        perform the oneshot encoding without iterating the for loop
        """
        # pad the input and reshape the data
        if data.dim() != 2:
            data = data.reshape(-1, data.shape[-1])
        self.data_device = data.device
        self.original_dtype = data.dtype
        self.original_data_shape = data.shape
        padded_data_len = (data.shape[1] + self.bvr_len - 1) // self.bvr_len * self.bvr_len
        self.padded_data_shape = (data.shape[0], padded_data_len)
        
        padded_data_diff = padded_data_len - data.shape[1]
        if padded_data_diff > 0:
            data = torch.cat([data, torch.zeros(data.shape[0], padded_data_diff, device=data.device, dtype=data.dtype)], dim=1)
        data = data.view(-1, 1, self.bvr_len, 1)
        groups = data.shape[0]
        
        # perform coeff searching
        candidate_matrix = self._get_candidate_mm_ext(data)
        n_ss_row = candidate_matrix.shape[0]
        n_ss_col = candidate_matrix.shape[1]
        candidate_matrix = candidate_matrix.view(1, n_ss_row, 1, n_ss_col)
        
        if self.data_device.type == "cuda":
            elem_size = torch.tensor([], dtype=self.original_dtype).element_size()
            diff_mat_size = (self.bvr_len * n_ss_row * n_ss_col * elem_size)
            free_mem = torch.cuda.mem_get_info(self.data_device)[0]
            diff_batch_size = max(1, int(free_mem * 0.1 // diff_mat_size))
            # print(b_str(f"diff_batch_size: {diff_batch_size}"))
            # sys.exit(0)
        else:
            raise ValueError(r_str("Only CUDA is supported for now"))
        
        if not use_oom_avoidance:
            diff = (data - candidate_matrix)**2 # diff.shape: (-1, n_ss_row, bvr_len, n_ss_col)
            diff_selected, coeff_comb_indices = diff.min(dim=-1) # diff_selected.shape: (-1, n_ss_row, bvr_len), coeff_comb_indices.shape: (-1, n_ss_row, bvr_len)
            mse = diff_selected.to(torch.float32).mean(dim=-1) # mse.shape: (-1, n_ss_row)
            
            min_indices = mse.argmin(dim=-1) # min_indices.shape: (-1,)
            
            group_indices = torch.arange(coeff_comb_indices.shape[0], device=self.data_device)
            
            coeff_comb_sel = coeff_comb_indices[group_indices, min_indices, :] # coeff_comb_sel.shape: (-1, bvr_len), use advanced indexing
            coeff_comb_sel = coeff_comb_sel.flatten()
        else:
            bs = min(diff_batch_size, groups)
            coeff_sel_chunks, coeff_idx_chunks = [], []
            for start in range(0, groups, bs):
                end = min(start + bs, groups)
                chunk = data[start:end]
                
                diff = (chunk - candidate_matrix)**2
                diff_selected, coeff_comb_indices = diff.min(dim=-1)
                mse = diff_selected.to(torch.float32).mean(dim=-1)
                
                min_indices = mse.argmin(dim=-1)
                group_indices = torch.arange(coeff_comb_indices.shape[0], device=self.data_device)
                coeff_comb_sel = coeff_comb_indices[group_indices, min_indices, :]
                
                coeff_sel_chunks.append(coeff_comb_sel)
                coeff_idx_chunks.append(min_indices)
            coeff_comb_sel = torch.cat(coeff_sel_chunks, dim=0).flatten()
            min_indices = torch.cat(coeff_idx_chunks, dim=0)
        
        self.coeff_sel = coeff_comb_sel.to(torch.int32).to(self.data_device)
        self.coeff_idx = min_indices.to(self.data_device)
        
        if self.coeff_set.shape[0] <= 256:
            self.coeff_idx = self.coeff_idx.to(torch.uint8)
        else:
            self.coeff_idx = self.coeff_idx.to(torch.uint16)
        
        # change the results to sbvr format
        bvr = self._change_coeff_sel_to_bvr()
        bvr = bvr.view(self.num_sums, -1, self.padded_data_shape[-1] // \
                                                self._get_bvr_num_bits())
        bvr = bvr.permute(2, 1, 0).contiguous()
        self.bvr = bvr
        
        coeff_idx = self.coeff_idx.view(-1, self.bvr.shape[0] // \
                                (self.bvr_len // self._get_bvr_num_bits()))
        coeff_idx = coeff_idx.transpose(0, 1).contiguous()
        self.coeff_idx = coeff_idx
        
        return self.bvr, self.coeff_idx
    
    @torch.inference_mode()
    def decode(self):
        decoded_tensor = torch.empty(self.padded_data_shape,
                                      dtype=self.original_dtype,
                                      device=self.data_device)
        num_bvr = self.coeff_idx.numel()
        coeff_sel = self._change_bvr_to_coeff_sel()
        coeff_idx = self.coeff_idx.transpose(0, 1).contiguous().flatten()
        for i in range(num_bvr):
            group_start = i * self.bvr_len
            group_end = \
                min(group_start + self.bvr_len, decoded_tensor.numel())
            group_coeff = self.coeff_set[coeff_idx[i].item()]
            group_coeff_sel = coeff_sel[group_start:group_end]
            group_all_points = self._get_all_points(group_coeff)
            group_data = group_all_points[group_coeff_sel]
            decoded_tensor.flatten()[group_start:group_end] = group_data
            
        # Truncate the tensor to the original shape
        # TODO: modify this part
        if self.original_data_shape != self.padded_data_shape:
            slices = tuple(slice(0, s) for s in self.original_data_shape)
            decoded_tensor = decoded_tensor[slices]
        
        return decoded_tensor
    
    def clean_encode_info(self, release_memory=False):
        self.coeff_sel = None
        self.coeff_idx = None
        self.bvr = None
        
        self.conversion_complete = False
        
        self.data_device = None
        self.original_dtype = None
        self.original_data_shape = None
        self.padded_data_shape = None
        
        if release_memory:
            cleanup_memory()
        