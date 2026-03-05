"""
Kernel Runner Module for SBVR Kernels
=====================================
This module provides utilities to run SBVR CUDA kernels with dummy data
for profiling with Nsight Compute (ncu).

Supported kernels:
1. fused_rtn7_lut_bvr - RTN quantization + LUT remap + BVR packing
2. rtn_7_sbvr_1xtN_mm_T - RTN-SBVR matrix multiplication kernel
"""

import torch
import argparse
from typing import Tuple, Optional

# Import SBVR CUDA operations
try:
    from sbvr.sbvr_cuda import (
        _sbvr_cuda_init,
        _sbvr_input_transfrom,
        _rtn_sbvr_1xtN_mm_T,
        _fused_rtn_sbvr_1xtN_mm_T,
    )
except ImportError as e:
    print(f"Error importing SBVR CUDA module: {e}")
    print("Make sure to install sbvr package first: pip install -e .")
    raise


# Constants from kernel implementation
GROUP_SIZE = 128
K_PER_BVR = 4


def create_dummy_input_transform_tensors(
    K: int = 4096,
    device: str = "cuda:0"
) -> Tuple[torch.Tensor]:
    """
    Create dummy tensors for fused_rtn7_lut_bvr kernel (via _sbvr_input_transfrom).
    
    This kernel performs:
    1. Group max-abs reduction
    2. RTN quantization + LUT remap
    3. 8x32-bit BVR packing
    
    Args:
        K: Hidden dimension size (must be multiple of 128)
        device: CUDA device
    
    Returns:
        Tuple of (x,) where:
        - x: (1, K) half-precision random input tensor
    """
    assert K % GROUP_SIZE == 0, f"K must be multiple of {GROUP_SIZE}"
    
    # Create random half-precision input
    x = torch.randn(1, K, dtype=torch.float16, device=device)
    
    return (x,)


def create_dummy_rtn_sbvr_mm_tensors(
    N: int = 4096,
    K: int = 4096,
    r_num_sums: int = 4,
    r_cache_size: int = 256,
    nRTN: int = 7,
    device: str = "cuda:0"
) -> Tuple[torch.Tensor, ...]:
    """
    Create dummy tensors for rtn_sbvr_1xtN_mm_T kernel.
    
    This kernel performs RTN-SBVR matrix multiplication:
    out[1, N] = l_bvr @ r_bvr^T (with RTN scaling and SBVR coefficients)
    
    Args:
        N: Output dimension (number of columns in weight matrix)
        K: Input dimension (number of rows, must be multiple of 128)
        r_num_sums: Number of SBVR sums (typically 4)
        r_cache_size: Coefficient cache size (256 for uint8 indices)
        nRTN: RTN bits (7, 5, or 3)
        device: CUDA device
    
    Returns:
        Tuple of tensors required by _rtn_sbvr_1xtN_mm_T:
        (l_bvr, l_scales, r_bvr, r_coeff_idx, r_coeff_cache, bias)
    """
    assert K % GROUP_SIZE == 0, f"K must be multiple of {GROUP_SIZE}"
    assert K % 32 == 0, "K must be multiple of 32"
    
    _nRTN = nRTN + 1  # Internal representation
    num_groups = K // GROUP_SIZE
    
    # Left side (input) tensors after RTN transformation
    # l_bvr: (K/32, _nRTN) uint32 - bit vectors for input
    l_bvr = torch.randint(
        0, 2**31, 
        (K // 32, _nRTN), 
        dtype=torch.uint32,  # will be reinterpreted as uint32
        device=device
    )
    
    # l_scales: (num_groups,) float32 - scales per group
    l_scales = torch.rand(num_groups, dtype=torch.float32, device=device) * 0.1 + 0.01
    
    # Right side (weight) tensors - SBVR quantized
    # r_bvr: (K/32, N, r_num_sums) uint32 - bit vectors for weight
    # Note: The kernel expects r_bvr in format (K, N, r_num_sums)
    r_bvr = torch.randint(
        0, 2**31,
        (K // 32, N, r_num_sums),
        dtype=torch.uint32,
        device=device
    )
    
    # r_coeff_idx: (K/128, N) - coefficient indices
    # Use uint8 if cache_size <= 256, else uint16
    idx_dtype = torch.uint8 if r_cache_size <= 256 else torch.uint16
    r_coeff_idx = torch.randint(
        0, r_cache_size,
        (K // 128, N),
        dtype=idx_dtype,
        device=device
    )
    
    # r_coeff_cache: (r_cache_size, r_num_sums) half - coefficient lookup table
    r_coeff_cache = torch.randn(
        r_cache_size, r_num_sums,
        dtype=torch.float16,
        device=device
    ) * 0.1
    
    # bias: (N,) half - optional bias (can be empty tensor for no bias)
    bias = torch.zeros(N, dtype=torch.float16, device=device)
    
    return (l_bvr, l_scales, r_bvr, r_coeff_idx, r_coeff_cache, bias)


def run_input_transform_kernel(
    x: torch.Tensor,
    nRTN: int = 7,
    group_size: int = 128,
    warmup: int = 10,
    iterations: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Run the fused RTN LUT BVR kernel.
    
    Args:
        x: Input tensor (1, K) half-precision
        nRTN: RTN bits (7, 5, or 3)
        group_size: Group size (default 128)
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
    
    Returns:
        Tuple of (out_bvr, scales, avg_time_ms)
    """
    # Squeeze to 1D if needed
    if x.dim() == 2:
        x = x.squeeze(0)
    
    # Warmup
    for _ in range(warmup):
        out_bvr, scales = _sbvr_input_transfrom(x, nRTN, group_size)
    
    torch.cuda.synchronize()
    
    # Timed iterations
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iterations):
        out_bvr, scales = _sbvr_input_transfrom(x, nRTN, group_size)
    end_event.record()
    
    torch.cuda.synchronize()
    avg_time_ms = start_event.elapsed_time(end_event) / iterations
    
    return out_bvr, scales, avg_time_ms


def run_rtn_sbvr_mm_kernel(
    l_bvr: torch.Tensor,
    l_scales: torch.Tensor,
    r_bvr: torch.Tensor,
    r_coeff_idx: torch.Tensor,
    r_coeff_cache: torch.Tensor,
    bias: torch.Tensor,
    nRTN: int = 7,
    warmup: int = 10,
    iterations: int = 100
) -> Tuple[torch.Tensor, float]:
    """
    Run the RTN-SBVR matrix multiplication kernel.
    
    Args:
        l_bvr: Left BVR tensor (K/32, _nRTN) uint32
        l_scales: Left scales tensor (num_groups,) float32
        r_bvr: Right BVR tensor (K, N, r_num_sums) uint32
        r_coeff_idx: Coefficient indices (K, N) uint8/uint16
        r_coeff_cache: Coefficient cache (cache_size, r_num_sums) half
        bias: Bias tensor (N,) half
        nRTN: RTN bits (7, 5, or 3)
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
    
    Returns:
        Tuple of (output, avg_time_ms)
    """
    # Warmup
    for _ in range(warmup):
        out = _rtn_sbvr_1xtN_mm_T(
            l_bvr, l_scales, r_bvr, r_coeff_idx, r_coeff_cache, bias, nRTN
        )
    
    torch.cuda.synchronize()
    
    # Timed iterations
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iterations):
        out = _rtn_sbvr_1xtN_mm_T(
            l_bvr, l_scales, r_bvr, r_coeff_idx, r_coeff_cache, bias, nRTN
        )
    end_event.record()
    
    torch.cuda.synchronize()
    avg_time_ms = start_event.elapsed_time(end_event) / iterations
    
    return out, avg_time_ms


def run_fused_rtn_sbvr_mm_kernel(
    x: torch.Tensor,
    r_bvr: torch.Tensor,
    r_coeff_idx: torch.Tensor,
    r_coeff_cache: torch.Tensor,
    bias: torch.Tensor,
    nRTN: int = 7,
    warmup: int = 10,
    iterations: int = 100
) -> Tuple[torch.Tensor, float]:
    """
    Run the fused RTN-SBVR kernel (input transform + matmul combined).
    
    Args:
        x: Input tensor (1, K) half-precision
        r_bvr: Right BVR tensor (K, N, r_num_sums) uint32
        r_coeff_idx: Coefficient indices (K, N) uint8/uint16
        r_coeff_cache: Coefficient cache (cache_size, r_num_sums) half
        bias: Bias tensor (N,) half
        nRTN: RTN bits (7, 5, or 3)
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
    
    Returns:
        Tuple of (output, avg_time_ms)
    """
    # Squeeze to 1D if needed
    if x.dim() == 2:
        x = x.squeeze(0)
    
    # Warmup
    for _ in range(warmup):
        out = _fused_rtn_sbvr_1xtN_mm_T(
            x, r_bvr, r_coeff_idx, r_coeff_cache, bias, nRTN
        )
    
    torch.cuda.synchronize()
    
    # Timed iterations
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iterations):
        out = _fused_rtn_sbvr_1xtN_mm_T(
            x, r_bvr, r_coeff_idx, r_coeff_cache, bias, nRTN
        )
    end_event.record()
    
    torch.cuda.synchronize()
    avg_time_ms = start_event.elapsed_time(end_event) / iterations
    
    return out, avg_time_ms


def profile_single_kernel_run(
    kernel_name: str,
    N: int = 4096,
    K: int = 4096,
    nRTN: int = 7,
    r_num_sums: int = 4,
    r_cache_size: int = 256,
    device: str = "cuda:0"
):
    """
    Run a single kernel invocation for NCU profiling.
    This function is called by the ncu profiler script.
    
    Args:
        kernel_name: "fused_rtn_lut_bvr", "rtn_sbvr_mm", or "fused_rtn_sbvr_mm"
        N: Output dimension
        K: Input/hidden dimension
        nRTN: RTN bits
        r_num_sums: Number of SBVR sums
        r_cache_size: Coefficient cache size
        device: CUDA device
    """
    torch.cuda.set_device(device)
    
    if kernel_name == "fused_rtn_lut_bvr":
        # Create dummy data
        (x,) = create_dummy_input_transform_tensors(K=K, device=device)
        x = x.squeeze(0)  # Make 1D
        
        # Single kernel run for profiling
        torch.cuda.synchronize()
        out_bvr, scales = _sbvr_input_transfrom(x, nRTN, GROUP_SIZE)
        torch.cuda.synchronize()
        
        print(f"✓ fused_rtn{nRTN}_lut_bvr executed successfully")
        print(f"  Input shape: {x.shape}, Output BVR shape: {out_bvr.shape}, Scales shape: {scales.shape}")
        
    elif kernel_name == "rtn_sbvr_mm":
        # Create dummy data
        (l_bvr, l_scales, r_bvr, r_coeff_idx, r_coeff_cache, bias) = \
            create_dummy_rtn_sbvr_mm_tensors(
                N=N, K=K, r_num_sums=r_num_sums, 
                r_cache_size=r_cache_size, nRTN=nRTN, device=device
            )
        
        # Single kernel run for profiling
        torch.cuda.synchronize()
        out = _rtn_sbvr_1xtN_mm_T(
            l_bvr, l_scales, r_bvr, r_coeff_idx, r_coeff_cache, bias, nRTN
        )
        torch.cuda.synchronize()
        
        print(f"✓ rtn_{nRTN}_sbvr_1xtN_mm_T executed successfully")
        print(f"  l_bvr: {l_bvr.shape}, r_bvr: {r_bvr.shape}, Output: {out.shape}")
        
    elif kernel_name == "fused_rtn_sbvr_mm":
        # Create input tensor
        x = torch.randn(K, dtype=torch.float16, device=device)
        
        # Create weight tensors (same as rtn_sbvr_mm)
        _, _, r_bvr, r_coeff_idx, r_coeff_cache, bias = \
            create_dummy_rtn_sbvr_mm_tensors(
                N=N, K=K, r_num_sums=r_num_sums,
                r_cache_size=r_cache_size, nRTN=nRTN, device=device
            )
        
        # Single kernel run for profiling
        torch.cuda.synchronize()
        out = _fused_rtn_sbvr_1xtN_mm_T(
            x, r_bvr, r_coeff_idx, r_coeff_cache, bias, nRTN
        )
        torch.cuda.synchronize()
        
        print(f"✓ fused_rtn_{nRTN}_sbvr_1xtN_mm_T executed successfully")
        print(f"  x: {x.shape}, r_bvr: {r_bvr.shape}, Output: {out.shape}")
        
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SBVR kernels for NCU profiling")
    parser.add_argument(
        "--kernel", "-k",
        type=str,
        required=True,
        choices=["fused_rtn_lut_bvr", "rtn_sbvr_mm", "fused_rtn_sbvr_mm"],
        help="Kernel to profile"
    )
    parser.add_argument("--N", type=int, default=4096, help="Output dimension")
    parser.add_argument("--K", type=int, default=4096, help="Input/hidden dimension")
    parser.add_argument("--nRTN", type=int, default=7, choices=[3, 5, 7], help="RTN bits")
    parser.add_argument("--r_num_sums", type=int, default=4, help="Number of SBVR sums")
    parser.add_argument("--r_cache_size", type=int, default=256, help="Coefficient cache size")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    
    args = parser.parse_args()
    
    profile_single_kernel_run(
        kernel_name=args.kernel,
        N=args.N,
        K=args.K,
        nRTN=args.nRTN,
        r_num_sums=args.r_num_sums,
        r_cache_size=args.r_cache_size,
        device=args.device
    )
