"""
Benchmark script for TPU MatMul Kernels

This script benchmarks all the matrix multiplication kernel implementations
and compares their performance with the native XLA implementation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import time
import argparse
from tabulate import tabulate

# Import all matmul kernel implementations
from src.kernels.matmul_v1 import run_matmul_v1
from src.kernels.matmul_v2_parallel import run_matmul_v2
from src.kernels.matmul_v3_block import run_matmul_v3
from src.kernels.matmul_v4_bf16 import run_matmul_v4
from src.kernels.matmul_v5_large_block import run_matmul_v5
from src.kernels.matmul_v6_int8 import run_matmul_v6
from src.kernels.matmul_v7_batch import batch_matmul_with_activation

# Import benchmark utilities
from src.utils.benchmark import get_matmul_performance, v5e_flops

def main():
    parser = argparse.ArgumentParser(description='Benchmark TPU MatMul kernels')
    parser.add_argument('--sizes', nargs='+', type=int, default=[1024, 4096, 8192],
                      help='Matrix sizes to benchmark (default: 1024 4096 8192)')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'],
                      help='Data type to use (default: bfloat16)')
    parser.add_argument('--trials', type=int, default=10,
                      help='Number of trials for each benchmark (default: 10)')
    args = parser.parse_args()
    
    # Set data type
    if args.dtype == 'float32':
        dtype = jnp.float32
    else:
        dtype = jnp.bfloat16
    
    print(f"Benchmarking TPU MatMul kernels with {args.dtype} precision")
    print(f"Running {args.trials} trials for each kernel")
    print()
    
    # Dictionary to store results
    results = {}
    
    # Run benchmarks for each matrix size
    for size in args.sizes:
        print(f"====== Matrix size: {size}x{size} ======")
        m, k, n = size, size, size
        
        # Parameters for parallel and block kernels
        N = 8  # Grid size for parallel kernel
        bm, bk, bn = 128, 128, 128  # Block sizes for standard block kernels
        bm_large, bk_large, bn_large = 512, 1024, 1024  # Block sizes for large block kernels
        
        # Define kernels to benchmark
        kernels = {
            "XLA MatMul": (jnp.matmul, {}),
            "V1: Naive MatMul": (run_matmul_v1, {}),
            "V2: Parallel MatMul": (run_matmul_v2, {"N": N}),
            "V3: Block MatMul": (run_matmul_v3, {"bm": bm, "bk": bk, "bn": bn}),
            "V4: BFloat16 MatMul": (run_matmul_v4, {"bm": bm, "bk": bk, "bn": bn}),
            "V5: Large Block MatMul": (run_matmul_v5, {"bm": bm_large, "bk": bk_large, "bn": bn_large}),
            "V6: INT8 MatMul": (run_matmul_v6, {"bm": bm_large, "bk": bk_large, "bn": bn_large}),
        }
        
        # Skip V1 and V2 for large matrices to avoid memory issues
        if size > 4096:
            del kernels["V1: Naive MatMul"]
            del kernels["V2: Parallel MatMul"]
        
        # Results for this size
        size_results = []
        
        # Run benchmarks
        for kernel_name, (kernel_func, kernel_kwargs) in kernels.items():
            print(f"Running {kernel_name}...")
            
            try:
                flops, time_taken = get_matmul_performance(m, k, n, dtype, kernel_func, **kernel_kwargs)
                gflops = flops / 1e9
                tpu_util = flops / v5e_flops * 100
                
                # Compare with XLA if not XLA itself
                if kernel_name != "XLA MatMul":
                    xla_flops, _ = get_matmul_performance(m, k, n, dtype, jnp.matmul)
                    xla_percentage = flops / xla_flops * 100
                else:
                    xla_percentage = 100.0
                
                size_results.append({
                    "Kernel": kernel_name,
                    "Time (ms)": time_taken * 1000,
                    "GFLOP/s": gflops,
                    "TPU Util %": tpu_util,
                    "% of XLA": xla_percentage
                })
                
                print(f"  Time: {time_taken*1000:.3f} ms, Performance: {gflops:.2f} GFLOP/s")
                print(f"  TPU Utilization: {tpu_util:.2f}%, XLA Performance: {xla_percentage:.2f}%")
                
            except Exception as e:
                print(f"  Failed: {e}")
                size_results.append({
                    "Kernel": kernel_name,
                    "Time (ms)": float('nan'),
                    "GFLOP/s": float('nan'),
                    "TPU Util %": float('nan'),
                    "% of XLA": float('nan')
                })
        
        results[size] = size_results
        print()
    
    # Print summary table
    print("=== Summary ===")
    for size, size_results in results.items():
        print(f"\nMatrix size: {size}x{size}")
        print(tabulate(size_results, headers="keys", tablefmt="grid", floatfmt=".2f"))

if __name__ == "__main__":
    main() 