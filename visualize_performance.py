"""
Visualization script for TPU MatMul Kernel performance

This script runs benchmarks and creates visualizations of the performance
of the different matrix multiplication kernels.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
from datetime import datetime

# Import all matmul kernel implementations
from src.kernels.matmul_v1 import run_matmul_v1
from src.kernels.matmul_v2_parallel import run_matmul_v2
from src.kernels.matmul_v3_block import run_matmul_v3
from src.kernels.matmul_v4_bf16 import run_matmul_v4
from src.kernels.matmul_v5_large_block import run_matmul_v5
from src.kernels.matmul_v6_int8 import run_matmul_v6

# Import benchmark utilities
from src.utils.benchmark import get_matmul_performance, v5e_flops

def run_benchmarks(sizes, dtype=jnp.bfloat16):
    """Run benchmarks for all kernel implementations and matrix sizes"""
    
    results = {}
    
    for size in sizes:
        print(f"Benchmarking matrices of size {size}x{size}")
        m, k, n = size, size, size
        
        # Parameters for parallel and block kernels
        N = 8  # Grid size for parallel kernel
        bm, bk, bn = 128, 128, 128  # Block sizes for standard block kernels
        bm_large, bk_large, bn_large = 512, 1024, 1024  # Block sizes for large block kernels
        
        # Define kernels to benchmark
        kernels = {
            "XLA MatMul": (jnp.matmul, {}),
            "V1: Naive": (run_matmul_v1, {}) if size <= 4096 else None,
            "V2: Parallel": (run_matmul_v2, {"N": N}) if size <= 4096 else None,
            "V3: Block": (run_matmul_v3, {"bm": bm, "bk": bk, "bn": bn}),
            "V4: BFloat16": (run_matmul_v4, {"bm": bm, "bk": bk, "bn": bn}),
            "V5: Large Block": (run_matmul_v5, {"bm": bm_large, "bk": bk_large, "bn": bn_large}),
            "V6: INT8": (run_matmul_v6, {"bm": bm_large, "bk": bk_large, "bn": bn_large}),
        }
        
        size_results = {}
        
        # Run benchmarks
        for kernel_name, kernel_info in kernels.items():
            if kernel_info is None:
                continue
                
            kernel_func, kernel_kwargs = kernel_info
            
            try:
                print(f"  Running {kernel_name}...")
                flops, time_taken = get_matmul_performance(m, k, n, dtype, kernel_func, **kernel_kwargs)
                gflops = flops / 1e9
                
                size_results[kernel_name] = {
                    "time_ms": time_taken * 1000,
                    "gflops": gflops,
                    "tpu_util_pct": flops / v5e_flops * 100,
                }
                
                # Calculate % of XLA performance
                if kernel_name != "XLA MatMul" and "XLA MatMul" in size_results:
                    xla_gflops = size_results["XLA MatMul"]["gflops"]
                    size_results[kernel_name]["xla_pct"] = gflops / xla_gflops * 100
                else:
                    size_results[kernel_name]["xla_pct"] = 100.0
                    
                print(f"    {gflops:.2f} GFLOP/s, {time_taken*1000:.2f} ms")
                
            except Exception as e:
                print(f"    Failed: {e}")
        
        results[size] = size_results
    
    return results

def plot_performance(results, output_dir="plots"):
    """Generate plots from benchmark results"""
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    sizes = sorted(list(results.keys()))
    kernel_names = []
    for size_results in results.values():
        for name in size_results.keys():
            if name not in kernel_names:
                kernel_names.append(name)
    
    # Remove XLA from kernel_names for some plots
    kernels_without_xla = [name for name in kernel_names if name != "XLA MatMul"]
    
    # Collect data for plotting
    performance_data = {name: [] for name in kernel_names}
    xla_pct_data = {name: [] for name in kernels_without_xla}
    
    for size in sizes:
        size_results = results[size]
        for name in kernel_names:
            if name in size_results:
                performance_data[name].append(size_results[name]["gflops"])
            else:
                performance_data[name].append(0)
        
        for name in kernels_without_xla:
            if name in size_results:
                xla_pct_data[name].append(size_results[name].get("xla_pct", 0))
            else:
                xla_pct_data[name].append(0)
    
    # 1. Plot raw performance (GFLOP/s)
    plt.figure(figsize=(12, 8))
    markers = ['o', 's', '^', 'D', 'v', '<', '>']
    for i, name in enumerate(kernel_names):
        plt.plot(sizes, performance_data[name], marker=markers[i % len(markers)], 
                 linewidth=2, markersize=8, label=name)
    
    plt.xscale('log', base=2)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Performance (GFLOP/s)', fontsize=14)
    plt.title('MatMul Kernel Performance by Matrix Size', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_{timestamp}.png", dpi=300)
    
    # 2. Plot percentage of XLA performance
    plt.figure(figsize=(12, 8))
    for i, name in enumerate(kernels_without_xla):
        plt.plot(sizes, xla_pct_data[name], marker=markers[i % len(markers)],
                 linewidth=2, markersize=8, label=name)
    
    plt.xscale('log', base=2)
    plt.xlabel('Matrix Size', fontsize=14)
    plt.ylabel('Percentage of XLA Performance (%)', fontsize=14)
    plt.title('MatMul Kernel Performance Relative to XLA', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=100, color='r', linestyle='--', alpha=0.6, label='XLA Baseline')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/xla_percentage_{timestamp}.png", dpi=300)
    
    # 3. Bar chart for the largest matrix size
    largest_size = max(sizes)
    largest_results = results[largest_size]
    
    plt.figure(figsize=(12, 8))
    names = []
    perf_values = []
    colors = plt.cm.viridis(np.linspace(0, 1, len(largest_results)))
    
    for i, (name, data) in enumerate(largest_results.items()):
        names.append(name)
        perf_values.append(data["gflops"])
    
    bars = plt.bar(names, perf_values, color=colors)
    plt.xlabel('Kernel Implementation', fontsize=14)
    plt.ylabel('Performance (GFLOP/s)', fontsize=14)
    plt.title(f'MatMul Kernel Performance for {largest_size}x{largest_size} Matrices', fontsize=16)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height+5,
                 f'{height:.1f}',
                 ha='center', va='bottom', rotation=0, fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/largest_size_comparison_{timestamp}.png", dpi=300)
    
    # Save results as JSON for later analysis
    with open(f"{output_dir}/benchmark_results_{timestamp}.json", 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    
    print(f"Plots saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Benchmark and visualize TPU MatMul kernels')
    parser.add_argument('--sizes', nargs='+', type=int, default=[1024, 2048, 4096, 8192],
                      help='Matrix sizes to benchmark (default: 1024 2048 4096 8192)')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16'],
                      help='Data type to use (default: bfloat16)')
    parser.add_argument('--output', type=str, default='plots',
                      help='Output directory for plots (default: plots)')
    args = parser.parse_args()
    
    # Set data type
    if args.dtype == 'float32':
        dtype = jnp.float32
    else:
        dtype = jnp.bfloat16
    
    print(f"Benchmarking TPU MatMul kernels with {args.dtype} precision")
    results = run_benchmarks(args.sizes, dtype)
    plot_performance(results, args.output)

if __name__ == "__main__":
    main()
