#!/usr/bin/env python3
"""
SBVR Kernel Profiling with Nsight Compute (NCU)
=================================================

This script provides two modes for analyzing SBVR CUDA kernels:

1. **Benchmark Mode**: Quick timing measurements using CUDA events
2. **NCU Profile Mode**: Detailed profiling using Nsight Compute

Supported Kernels:
- fused_rtn7_lut_bvr: RTN quantization + LUT remap + BVR packing
- rtn_7_sbvr_1xtN_mm_T: RTN-SBVR matrix multiplication

Usage:
------
# Benchmark mode (quick timing)
python eval_sbvr_kernels.py --mode benchmark --kernel all

# Generate NCU commands (for manual profiling)
python eval_sbvr_kernels.py --mode ncu --kernel rtn_sbvr_mm

# Full NCU profiling (requires sudo/root)
sudo python eval_sbvr_kernels.py --mode ncu --kernel all --run-ncu
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# Import kernel runner
from kernel_runner import (
    create_dummy_input_transform_tensors,
    create_dummy_rtn_sbvr_mm_tensors,
    run_input_transform_kernel,
    run_rtn_sbvr_mm_kernel,
    run_fused_rtn_sbvr_mm_kernel,
    profile_single_kernel_run,
)


# Default test configurations
DEFAULT_CONFIGS = {
    "small": {"N": 2048, "K": 2048},
    "medium": {"N": 4096, "K": 4096},
    "large": {"N": 8192, "K": 8192},
    "llama_7b_attn": {"N": 4096, "K": 4096},       # Self-attention projection
    "llama_7b_ffn_up": {"N": 11008, "K": 4096},    # FFN up projection
    "llama_7b_ffn_down": {"N": 4096, "K": 11008},  # FFN down projection
}


def run_benchmark(
    kernel_names: List[str],
    configs: List[str],
    nRTN_values: List[int],
    r_num_sums: int = 4,
    r_cache_size: int = 256,
    warmup: int = 10,
    iterations: int = 100,
    device: str = "cuda:0"
) -> Dict[str, Any]:
    """
    Run benchmark mode: measure kernel execution times using CUDA events.
    
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "device": torch.cuda.get_device_name(device),
        "benchmarks": []
    }
    
    print("=" * 70)
    print("SBVR Kernel Benchmark")
    print("=" * 70)
    print(f"Device: {results['device']}")
    print(f"Warmup iterations: {warmup}")
    print(f"Timed iterations: {iterations}")
    print()
    
    for config_name in configs:
        config = DEFAULT_CONFIGS[config_name]
        N, K = config["N"], config["K"]
        
        for nRTN in nRTN_values:
            print(f"\n--- Config: {config_name} (N={N}, K={K}), nRTN={nRTN} ---")
            
            for kernel_name in kernel_names:
                try:
                    if kernel_name == "fused_rtn_lut_bvr":
                        # Create dummy data
                        (x,) = create_dummy_input_transform_tensors(K=K, device=device)
                        
                        # Run benchmark
                        out_bvr, scales, avg_time = run_input_transform_kernel(
                            x, nRTN=nRTN, group_size=128,
                            warmup=warmup, iterations=iterations
                        )
                        
                        # Calculate throughput
                        bytes_read = K * 2  # fp16 input
                        bytes_written = (K // 32 * (nRTN + 1) * 4) + (K // 128 * 4)  # BVR + scales
                        bandwidth_gbps = (bytes_read + bytes_written) / (avg_time * 1e-3) / 1e9
                        
                        result = {
                            "kernel": kernel_name,
                            "config": config_name,
                            "N": N, "K": K, "nRTN": nRTN,
                            "avg_time_ms": avg_time,
                            "bandwidth_gbps": bandwidth_gbps,
                        }
                        
                    elif kernel_name == "rtn_sbvr_mm":
                        # Create dummy data
                        (l_bvr, l_scales, r_bvr, r_coeff_idx, r_coeff_cache, bias) = \
                            create_dummy_rtn_sbvr_mm_tensors(
                                N=N, K=K, r_num_sums=r_num_sums,
                                r_cache_size=r_cache_size, nRTN=nRTN, device=device
                            )
                        
                        # Run benchmark
                        out, avg_time = run_rtn_sbvr_mm_kernel(
                            l_bvr, l_scales, r_bvr, r_coeff_idx, r_coeff_cache, bias,
                            nRTN=nRTN, warmup=warmup, iterations=iterations
                        )
                        
                        # Calculate effective TOPS
                        ops = 2 * N * K  # Simple multiply-accumulate count
                        tops = ops / (avg_time * 1e-3) / 1e12
                        
                        result = {
                            "kernel": kernel_name,
                            "config": config_name,
                            "N": N, "K": K, "nRTN": nRTN,
                            "r_num_sums": r_num_sums,
                            "avg_time_ms": avg_time,
                            "effective_tops": tops,
                        }
                        
                    elif kernel_name == "fused_rtn_sbvr_mm":
                        # Create input tensor
                        x = torch.randn(K, dtype=torch.float16, device=device)
                        
                        # Create weight tensors
                        _, _, r_bvr, r_coeff_idx, r_coeff_cache, bias = \
                            create_dummy_rtn_sbvr_mm_tensors(
                                N=N, K=K, r_num_sums=r_num_sums,
                                r_cache_size=r_cache_size, nRTN=nRTN, device=device
                            )
                        
                        # Run benchmark
                        out, avg_time = run_fused_rtn_sbvr_mm_kernel(
                            x, r_bvr, r_coeff_idx, r_coeff_cache, bias,
                            nRTN=nRTN, warmup=warmup, iterations=iterations
                        )
                        
                        # Calculate effective TOPS
                        ops = 2 * N * K
                        tops = ops / (avg_time * 1e-3) / 1e12
                        
                        result = {
                            "kernel": kernel_name,
                            "config": config_name,
                            "N": N, "K": K, "nRTN": nRTN,
                            "r_num_sums": r_num_sums,
                            "avg_time_ms": avg_time,
                            "effective_tops": tops,
                        }
                        
                    else:
                        print(f"Unknown kernel: {kernel_name}")
                        continue
                    
                    results["benchmarks"].append(result)
                    
                    # Print result
                    if "bandwidth_gbps" in result:
                        print(f"  {kernel_name}: {avg_time:.4f} ms, {result['bandwidth_gbps']:.2f} GB/s")
                    else:
                        print(f"  {kernel_name}: {avg_time:.4f} ms, {result['effective_tops']:.3f} TOPS")
                        
                except Exception as e:
                    print(f"  {kernel_name}: ERROR - {e}")
                    results["benchmarks"].append({
                        "kernel": kernel_name,
                        "config": config_name,
                        "N": N, "K": K, "nRTN": nRTN,
                        "error": str(e)
                    })
    
    return results


def generate_ncu_commands(
    kernel_names: List[str],
    configs: List[str],
    nRTN_values: List[int],
    output_dir: str = "./ncu_reports",
    metrics_set: str = "full",
    python_executable: str = "/home/wjbang/anaconda3/envs/sbvr_nc/bin/python"
) -> List[str]:
    """
    Generate Nsight Compute (ncu) commands for kernel profiling.
    
    Args:
        kernel_names: List of kernel names to profile
        configs: List of configuration names
        nRTN_values: List of nRTN values
        output_dir: Directory for NCU reports
        metrics_set: "full", "roofline", or "summary"
        python_executable: Path to Python executable
    
    Returns:
        List of ncu command strings
    """
    script_path = Path(__file__).parent / "kernel_runner.py"
    
    # Metrics sets
    metric_options = {
        "full": "--set full",
        "roofline": "--set roofline",
        "summary": "--set summary",
        "memory": "--metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sectors.sum,dram__bytes.sum",
        "compute": "--metrics sm__sass_thread_inst_executed_op_integer_pred_on.sum,sm__sass_thread_inst_executed_op_fp16_pred_on.sum",
    }
    
    commands = []
    
    for config_name in configs:
        config = DEFAULT_CONFIGS[config_name]
        N, K = config["N"], config["K"]
        
        for nRTN in nRTN_values:
            for kernel_name in kernel_names:
                # Map kernel name for filtering
                kernel_filter_map = {
                    "fused_rtn_lut_bvr": f"fused_rtn{nRTN}_lut_bvr",
                    "rtn_sbvr_mm": f"rtn_{nRTN}_sbvr_1xtN_mm_T",
                    "fused_rtn_sbvr_mm": f"rtn_{nRTN}_sbvr_1xtN_mm_T",  # Contains both kernels
                }
                
                kernel_filter = kernel_filter_map.get(kernel_name, kernel_name)
                report_name = f"{kernel_name}_{config_name}_nRTN{nRTN}"
                report_path = Path(output_dir) / f"{report_name}.ncu-rep"
                
                cmd = [
                    "ncu",
                    metric_options.get(metrics_set, metrics_set),
                    f"--kernel-name-base function",
                    f"--kernel-name regex:{kernel_filter}",
                    f"-o {report_path}",
                    "--force-overwrite",
                    "--target-processes all",
                    python_executable,
                    str(script_path),
                    f"--kernel {kernel_name}",
                    f"--N {N}",
                    f"--K {K}",
                    f"--nRTN {nRTN}",
                ]
                
                commands.append(" ".join(cmd))
    
    return commands


def run_ncu_profiling(
    commands: List[str],
    output_dir: str = "./ncu_reports",
    dry_run: bool = False
) -> None:
    """
    Execute NCU profiling commands.
    
    Args:
        commands: List of ncu commands
        output_dir: Directory for reports
        dry_run: If True, only print commands without executing
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Nsight Compute Profiling")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Number of profiles: {len(commands)}")
    print()
    
    for i, cmd in enumerate(commands, 1):
        print(f"[{i}/{len(commands)}] Running:")
        print(f"  {cmd}")
        
        if dry_run:
            print("  (dry run - skipped)")
        else:
            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    print(f"  ERROR: {result.stderr}")
                else:
                    print("  SUCCESS")
                    
            except Exception as e:
                print(f"  ERROR: {e}")
        
        print()


def print_ncu_analysis_help():
    """Print help on how to analyze NCU reports."""
    help_text = """
================================================================================
How to Analyze NCU Reports
================================================================================

1. View in NCU GUI (Recommended):
   $ ncu-ui ./ncu_reports/kernel_name.ncu-rep

2. Export to CSV:
   $ ncu --import ./ncu_reports/kernel_name.ncu-rep --csv > analysis.csv

3. Print summary to console:
   $ ncu --import ./ncu_reports/kernel_name.ncu-rep --page details

4. Key Metrics to Look For:
   - Memory Throughput: How efficiently data is moved
   - Compute Throughput: Utilization of compute units
   - Achieved Occupancy: SM utilization
   - Warp Execution Efficiency: Thread divergence
   - Memory Bound vs Compute Bound analysis

5. For fused_rtn7_lut_bvr kernel:
   - Focus on memory bandwidth utilization
   - Check for bank conflicts in shared memory
   - Verify coalesced memory accesses

6. For rtn_sbvr_1xtN_mm_T kernel:
   - Check instruction throughput (popc operations)
   - Look at memory/compute balance
   - Analyze register pressure

================================================================================
"""
    print(help_text)


def main():
    parser = argparse.ArgumentParser(
        description="SBVR Kernel Profiling with Nsight Compute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick benchmark
  python eval_sbvr_kernels.py --mode benchmark --kernel all

  # Generate NCU commands
  python eval_sbvr_kernels.py --mode ncu --kernel rtn_sbvr_mm --dry-run

  # Full NCU profiling (requires sudo)
  sudo python eval_sbvr_kernels.py --mode ncu --kernel all --run-ncu

  # Profile specific configuration
  python eval_sbvr_kernels.py --mode benchmark --kernel rtn_sbvr_mm --config llama_7b_ffn_up
        """
    )
    
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["benchmark", "ncu", "help"],
        default="benchmark",
        help="Profiling mode"
    )
    
    parser.add_argument(
        "--kernel", "-k",
        type=str,
        nargs="+",
        default=["all"],
        help="Kernel(s) to profile: fused_rtn_lut_bvr, rtn_sbvr_mm, fused_rtn_sbvr_mm, or 'all'"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        nargs="+",
        default=["medium"],
        help=f"Configuration(s): {', '.join(DEFAULT_CONFIGS.keys())}, or 'all'"
    )
    
    parser.add_argument(
        "--nRTN",
        type=int,
        nargs="+",
        default=[7],
        choices=[3, 5, 7],
        help="RTN bit width(s)"
    )
    
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations for benchmark mode"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Timed iterations for benchmark mode"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ncu_reports",
        help="Output directory for NCU reports"
    )
    
    parser.add_argument(
        "--metrics-set",
        type=str,
        default="full",
        choices=["full", "roofline", "summary", "memory", "compute"],
        help="NCU metrics set"
    )
    
    parser.add_argument(
        "--run-ncu",
        action="store_true",
        help="Actually run NCU profiling (requires sudo)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device"
    )
    
    parser.add_argument(
        "--save-results",
        type=str,
        default=None,
        help="Save benchmark results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Expand 'all' options
    all_kernels = ["fused_rtn_lut_bvr", "rtn_sbvr_mm", "fused_rtn_sbvr_mm"]
    all_configs = list(DEFAULT_CONFIGS.keys())
    
    kernel_names = all_kernels if "all" in args.kernel else args.kernel
    configs = all_configs if "all" in args.config else args.config
    
    # Validate configs
    for c in configs:
        if c not in DEFAULT_CONFIGS:
            print(f"Error: Unknown config '{c}'. Available: {', '.join(DEFAULT_CONFIGS.keys())}")
            sys.exit(1)
    
    if args.mode == "help":
        print_ncu_analysis_help()
        return
    
    elif args.mode == "benchmark":
        results = run_benchmark(
            kernel_names=kernel_names,
            configs=configs,
            nRTN_values=args.nRTN,
            warmup=args.warmup,
            iterations=args.iterations,
            device=args.device
        )
        
        if args.save_results:
            with open(args.save_results, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.save_results}")
    
    elif args.mode == "ncu":
        commands = generate_ncu_commands(
            kernel_names=kernel_names,
            configs=configs,
            nRTN_values=args.nRTN,
            output_dir=args.output_dir,
            metrics_set=args.metrics_set
        )
        
        if args.run_ncu and not args.dry_run:
            run_ncu_profiling(commands, args.output_dir)
            print_ncu_analysis_help()
        else:
            print("=" * 70)
            print("Generated NCU Commands")
            print("=" * 70)
            print("Run these commands manually or use --run-ncu flag:")
            print()
            for cmd in commands:
                print(cmd)
                print()
            
            if not args.run_ncu:
                print("Note: Add --run-ncu to execute these commands")
                print("      NCU typically requires sudo/root permissions")


if __name__ == "__main__":
    main()
