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
import dataclasses

# Import all matmul kernel implementations
from src.kernels.matmul_v1 import run_matmul_v1
from src.kernels.matmul_v2_parallel import run_matmul_v2
from src.kernels.matmul_v3_block import run_matmul_v3
from src.kernels.matmul_v4_optimal_block_size import run_matmul_v4
from src.kernels.matmul_v5_quant_prec import run_matmul_v5

# Import benchmark utilities
from src.utils.benchmark import get_matmul_performance, v5e_flops


@dataclasses.dataclass
class KernelInfo:
  name: str
  func: callable
  kwargs: dict

_BASELINE_KERNEL_NAME = "Baseline XLA MatMul"

def run_benchmarks(sizes: list[int] = [512, 1024, 2048, 4096, 8192], kernel_selection: int = 0,
                   n_parallel: int = 8, bm: int = 128, bk: int = 128, bn: int = 128,
                   dtype=jnp.bfloat16) -> dict:
  """Run benchmarks for one kernel selection for different matrix sizes."""
  # Store results for each size: { size: {results}}
  results = {}

  for size in sizes:
    print(f"Benchmarking matrices of size {size}x{size}")
    m, k, n = size, size, size

    kernel_selection_map = {
        0: KernelInfo(name="XLA MatMul", func=jnp.matmul, kwargs={}),
        1: KernelInfo(name="V1: Naive", func=run_matmul_v1, kwargs={}),
        2: KernelInfo(name="V2: Parallel", func=run_matmul_v2, kwargs={"N": n_parallel}),
        3: KernelInfo(name="V3: Block", func=run_matmul_v3, kwargs={"bm": bm, "bk": bk, "bn": bn}),
        4: KernelInfo(name="V4: Optimal block size", func=run_matmul_v4, kwargs={"bm": bm, "bk": bk, "bn": bn}),
        5: KernelInfo(name="V5: quantization", func=run_matmul_v5, kwargs={"bm": bm, "bk": bk, "bn": bn}),
    }

    size_results = {}

    kernel = kernel_selection_map.get(kernel_selection, None)

    if kernel is None:
      print(f"Did not find kernel for selection {kernel_selection}")
      return None

    kernel_name = kernel.name
    kernel_func = kernel.func
    kernel_kwargs = kernel.kwargs

    try:
      print(f"  Running {kernel_name}...")
      flops, time_taken = get_matmul_performance(
        m, k, n, dtype, kernel_func, **kernel_kwargs)
      gflops = flops / 1e9

      size_results[kernel_name] = {
          "kernel": kernel_name,
          "M": m,
          "K": k,
          "N": n,
          "n_parallel": n_parallel,
          "bm": bm,
          "bk": bk,
          "bn": bn,
          "dtype": str(dtype),
          "gflops": gflops,
          "time_ms": time_taken * 1000,
          "tpu_util_pct": flops / v5e_flops * 100,
          "xla_pct": 0.0,
      }

      # Add XLA performance.
      if kernel_name == "XLA MatMul":
        size_results[kernel_name]["xla_pct"] = 100.0

      print(f"    {gflops:.2f} GFLOP/s, {time_taken * 1000:.2f} ms")

    except Exception as e:
      print(f"    Failed: {e}")

    results[size] = size_results

  print('Results:', results)
  return results


def plot_performance(results, baseline, output_dir="plots"):
  """Generate plots from benchmark results"""

  os.makedirs(output_dir, exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

  # All sizes
  sizes = sorted(list(baseline.keys()))
  # Get all kernel names from results
  all_kernel_names = [_BASELINE_KERNEL_NAME]
  custom_kernel_names = []
  for size_results in results.values():
    for name in size_results.keys():
      if name not in custom_kernel_names:
        custom_kernel_names.append(name)
  all_kernel_names.extend(custom_kernel_names)

  # Collect data for plotting
  performance_data = {name: [] for name in all_kernel_names}

  for size in sizes:
    # Get baseline performance
    baseline_results = baseline[size]
    performance_data[_BASELINE_KERNEL_NAME].append(
      baseline_results["XLA MatMul"]["gflops"])
    # Get custom kernel performance.
    custom_results = results[size]
    for name in custom_kernel_names:
      if name in custom_results:
        performance_data[name].append(custom_results[name]["gflops"])
      else:
        performance_data[name].append(0)

  # Plot raw performance (GFLOP/s)
  plt.figure(figsize=(12, 8))
  markers = ['o', 's', '^', 'D', 'v', '<', '>']
  for i, name in enumerate(all_kernel_names):
    plt.plot(sizes, performance_data[name], marker=markers[i % len(markers)],
             linewidth=2, markersize=8, label=name)

  # plt.xscale('log', base=2)
  plt.xlabel('Matrix Size', fontsize=14)
  plt.ylabel('Performance (GFLOP/s)', fontsize=14)
  plt.title('MatMul Kernel Performance by Matrix Size', fontsize=16)
  plt.grid(True, alpha=0.3)
  plt.legend(fontsize=12)
  plt.tight_layout()
  plt.savefig(f"{output_dir}/performance_{timestamp}.png", dpi=300)

  # Save results as JSON for later analysis
  with open(f"{output_dir}/benchmark_results_{timestamp}.json", 'w') as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)

  print(f"Plots saved to {output_dir}/")


def main():
  parser = argparse.ArgumentParser(
    description='Benchmark and visualize TPU MatMul kernels')
  parser.add_argument('--sizes', nargs='+', type=int, default=[512, 1024, 2048, 4096, 8192],
                      help='Matrix sizes to benchmark (default: 512, 1024 2048 4096 8192)')
  parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16', 'int8'],
                      help='Data type to use (default: bfloat16)')
  parser.add_argument('--output', type=str, default='plots',
                      help='Output directory for plots (default: plots)')
  parser.add_argument('--kernel', type=int, default=1,
                      choices=[0, 1, 2, 3, 4, 5],
                      help='Kernel selection (default: 1)')
  args = parser.parse_args()

  # Set data type
  if args.dtype == 'float32':
    dtype = jnp.float32
  elif args.dtype == 'bfloat16':
    dtype = jnp.bfloat16
  elif args.dtype == 'int8':
    dtype = jnp.int8
  else:
    raise ValueError(
      "Unsupported data type. Use 'float32', 'bfloat16', or 'int8'.")

  print(f"Benchmarking TPU MatMul kernels with {dtype} precision")
  baseline_xla_perf = run_benchmarks(sizes=args.sizes, kernel_selection=0,
                                     dtype=dtype)
  results = run_benchmarks(sizes=args.sizes, kernel_selection=args.kernel,
                           dtype=dtype)
  plot_performance(results, baseline_xla_perf, args.output)


if __name__ == "__main__":
  main()
