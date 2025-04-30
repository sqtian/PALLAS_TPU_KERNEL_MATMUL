# PALLAS TPU MatMul Kernels

This repository demonstrates various matrix multiplication kernel implementations for TPUs using JAX's Pallas framework. The kernels showcase different optimization strategies for matrix multiplication on TPU hardware.

## Overview

Matrix multiplication is a fundamental operation in many machine learning workloads. This repository explores different approaches to implementing and optimizing matrix multiplication kernels on TPUs using JAX's Pallas, a low-level programming model for accelerators.

## Set up a TPU machine

```bash
# Create a TPU VM on Google Cloud, and SSH into the instance.
$ ./create_tpu_then_ssh.sh

# Clone the repository
tpu_vm $ git clone https://github.com/sqtian/PALLAS_TPU_KERNEL_MATMUL.git
tpu_vm $ cd PALLAS_TPU_KERNEL_MATMUL

# Install dependencies
tpu_vm $ pip install -r requirements.txt
tpu_vm $ pip install -U "jax[tpu]>=0.4.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

```

## Kernel Implementations

This repository includes several MatMul kernel implementations, each showcasing different optimization techniques:

1. **V1: Naive MatMul**: A basic implementation without specific optimizations.
2. **V2: Parallel MatMul**: Splits matrices by rows/columns for parallel processing.
3. **V3: Block MatMul**: Implements block-based matrix multiplication using a 3D grid.
4. **V4: BFloat16 MatMul**: Utilizes BFloat16 precision with float32 accumulation for better performance.
5. **V5: Large Block MatMul**: Uses larger block sizes to optimize memory access patterns.
6. **V6: INT8 MatMul**: Implements quantized matrix multiplication using INT8 precision.
7. **V7: Batch MatMul**: Processes batched matrix multiplications with activation functions.

## Performance Comparison

Performance metrics for different kernel implementations on matrix dimensions 8192x8192:

| Kernel Implementation | GFLOP/s | % of XLA Performance |
|-----------------------|---------|----------------------|
| XLA MatMul            | ~X      | 100%                |
| V1: Naive MatMul      | ~Y      | ~Y%                 |
| V2: Parallel MatMul   | ~Z      | ~Z%                 |
| V3: Block MatMul      | ~A      | ~A%                 |
| V4: BFloat16 MatMul   | ~B      | ~B%                 |
| V5: Large Block MatMul| ~C      | ~C%                 |
| V6: INT8 MatMul       | ~D      | ~D%                 |

## Usage

```python
import jax
import jax.numpy as jnp
from src.kernels.matmul_v4_bf16 import run_matmul_v4

# Create input matrices
m, k, n = 4096, 4096, 4096
a = jnp.ones((m, k), dtype=jnp.bfloat16)
b = jnp.ones((k, n), dtype=jnp.bfloat16)

# Run the kernel
result = run_matmul_v4(a, b, bm=128, bk=128, bn=128)
```

## Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run a specific test
python -m unittest tests.test_matmul_v4_bf16
```

## Benchmarking

The repository includes scripts for benchmarking and visualizing the performance of different kernel implementations:

```bash
# Run benchmarks for all kernel implementations
python benchmark_all.py

# Generate performance plots
python visualize_performance.py
```

## License

MIT

## Acknowledgements

This project is inspired by JAX's Pallas framework and builds upon the TPU programming model.
