import torch
import itertools
import math
import numpy as np
from tqdm import tqdm

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
    
class sbvr_serialized():
    def __init__(self, 
                 num_sums: int,
                 bvr_len: int,
                 compute_dtype: torch.dtype,
                 bvr_dtype: torch.dtype,
                 original_dtype: torch.dtype,
                 original_data_shape: tuple,
                 bvr: torch.Tensor,
                 coeff_idx: torch.Tensor,
                 coeff_cache: torch.Tensor,
                 input_num_sums: int,
                 input_coeff: torch.Tensor):
        # Save base parameters
        self.num_sums = num_sums
        self.bvr_len = bvr_len
        self.compute_dtype = compute_dtype
        self.bvr_dtype = bvr_dtype
        bvr_num_bits = \
            torch.tensor(0, dtype=self.bvr_dtype).element_size() * 8
        if num_sums > 11 and self.compute_dtype == torch.float16:
            raise UserWarning(
                r_str("Warning: compute_dtype float16 does not have sufficient"
                      " precision for num_sums > 11."))
        if self.bvr_len % bvr_num_bits != 0:
            raise ValueError(
                r_str("BVR length must be a multiple of ") +
                      f"{bvr_num_bits}")
            
        self.original_dtype = original_dtype
        self.original_data_shape = original_data_shape
        self.padded_data_shape = list(self.original_data_shape)
        self.padded_data_shape[-1] = \
            (self.original_data_shape[-1] + self.bvr_len - 1) // \
            self.bvr_len * self.bvr_len
            
        if bvr.dtype != self.bvr_dtype:
            raise ValueError(
                r_str(f"The BVR data type does not match - expected type " +
                      f"{self.bvr_dtype} but got {bvr.dtype}"))
        if bvr.shape[2] != num_sums:
            raise ValueError(
                r_str("The number of summations does not match the BVR, " +
                      f"expected {num_sums} but got " + 
                      f"{bvr.shape[2]}"))
        if bvr.shape[0] * bvr_num_bits != self.padded_data_shape[-1]:
            raise ValueError(
                r_str("The BVR inner dimension does not match the padded "+
                      f"data shape, expected {self.padded_data_shape[-1]} " +
                      f"but got {bvr.shape[0] * bvr_num_bits}"))
        self.bvr = self._serialize_tensor(bvr)
        self.bvr_shape = bvr.shape
        self.bvr_dtype = bvr.dtype
        
        if coeff_cache.shape[0] <= 256:
            if coeff_idx.dtype != torch.uint8:
                raise ValueError(
                    r_str("The coefficient index data type must be uint8 " +
                            f"but got {coeff_idx.dtype}, " +
                            f"number of cache lines: {coeff_cache.shape[0]}"))
        elif coeff_cache.shape[0] <= 65536 and coeff_idx.dtype != torch.uint16:
            raise ValueError(
                r_str("The coefficient index data type must be uint16 " +
                        f"but got {coeff_idx.dtype}, " +
                        f"number of cache lines: {coeff_cache.shape[0]}"))
        elif coeff_cache.shape[0] > 65536:
            raise ValueError(
                r_str("Unsupported number of cache lines, " +
                        f"{coeff_cache.shape[0]}"))
        self.coeff_idx = self._serialize_tensor(coeff_idx)
        self.coeff_idx_shape = coeff_idx.shape
        self.coeff_idx_dtype = coeff_idx.dtype
        
        if coeff_cache.dtype != self.compute_dtype:
            raise ValueError(
                r_str("The coefficient cache data type does not match - "
                        f"expected type {self.compute_dtype} but got " +
                        f"{coeff_cache.dtype}"))
        self.coeff_cache = self._serialize_tensor(coeff_cache)
        self.coeff_cache_shape = coeff_cache.shape
        self.coeff_cache_dtype = coeff_cache.dtype
        
        self.input_num_sums = input_num_sums
        if input_coeff is not None:
            if input_coeff.dtype != self.compute_dtype:
                raise ValueError(
                    r_str("The input coefficient data type does not match - "
                            f"expected type {self.compute_dtype} but got " +
                            f"{input_coeff.dtype}"))
            self.input_coeff = self._serialize_tensor(input_coeff)
            self.input_coeff_shape = input_coeff.shape
            self.input_coeff_dtype = input_coeff.dtype
        else:
            self.input_coeff = None
            self.input_coeff_shape = None
            self.input_coeff_dtype = None
        
    def _serialize_tensor(self, tensor: torch.Tensor) -> bytes:
        raw_bytes = tensor.detach().cpu().numpy().tobytes()
        nd_array = np.frombuffer(raw_bytes, dtype=np.int8).copy()
        torch_tensor = torch.from_numpy(nd_array)
        return torch_tensor
    
    def _deserialize_tensor(self, serialized_data: torch.Tensor,
                            shape, dtype) -> torch.Tensor:
        serialized_data = serialized_data.detach().cpu().numpy()
        dtype_map = {
            "torch.uint8": np.uint8,
            "torch.uint16": np.uint16,
            "torch.uint32": np.uint32,
            "torch.int32": np.int32,
            "torch.float16": np.float16,
            "torch.float32": np.float32,
        }
        np_dtype = dtype_map.get(str(dtype), None)
        if np_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype}")
        array = \
            np.frombuffer(serialized_data, 
                          dtype=np_dtype).reshape(shape).copy()
        return torch.from_numpy(array).to(dtype=dtype).contiguous()
    
    def deserialize_sbvr(self):
        bvr = self._deserialize_tensor(self.bvr, self.bvr_shape, self.bvr_dtype)
        coeff_idx = self._deserialize_tensor(self.coeff_idx, 
                                             self.coeff_idx_shape, 
                                             self.coeff_idx_dtype)
        coeff_cache = self._deserialize_tensor(self.coeff_cache, 
                                               self.coeff_cache_shape, 
                                               self.coeff_cache_dtype)
        if self.input_coeff is not None:
            input_coeff = self._deserialize_tensor(self.input_coeff, 
                                                   self.input_coeff_shape, 
                                                   self.input_coeff_dtype)
        else:
            input_coeff = None
            
        return self.num_sums, self.bvr_len, self.compute_dtype, \
            self.bvr_dtype, self.original_dtype, self.original_data_shape, \
            bvr, coeff_idx, coeff_cache, self.input_num_sums, input_coeff

        