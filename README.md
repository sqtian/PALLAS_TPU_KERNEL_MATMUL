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
tpu_vm $ python -m pip install --upgrade pip
tpu_vm $ python -m pip install -r requirements.txt
tpu_vm $ python -m pip install -U "jax[tpu]>=0.4.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

## Run TPU Kernels
```bash
tpu_vm $ python src/kernels/matmul_v1.py 
tpu_vm $ python src/kernels/matmul_v2_parallel.py 
tpu_vm $ python src/kernels/matmul_v3_block.py 
tpu_vm $ python src/kernels/matmul_v4_optimal_block_size.py
tpu_vm $ python src/kernels/matmul_v5_quant_prec.py
```

## Kernel Implementations

This repository includes several MatMul kernel implementations, each showcasing different optimization techniques:

1. **V1: Naive MatMul**: A basic implementation without specific optimizations.
2. **V2: Parallel MatMul**: Splits matrices by rows/columns for parallel processing.
3. **V3: Block MatMul**: Implements block-based matrix multiplication using a 3D grid.
4. **V4: Optimal block size MatMul**: Uses larger block sizes to optimize memory access patterns. 
5. **V5: Quantization**: Utilizes BFloat16 or INT8 precision with float32 accumulation for better performance.

## Benchmarking

The repository includes scripts for benchmarking and visualizing the performance of different kernel implementations:

```bash
# Generate performance plots
tpu_vm $ python visualize_performance.py --analyze=5 --dtype="bfloat16"
```

## Performance

### Kernel 1

![Kernel 1 Performance](plots/performance_kernel_1.png)

### Kernel 2

![Kernel 2 Performance](plots/performance_kernel_2.png)

### Kernel 3

![Kernel 3 Performance](plots/performance_kernel_3.png)

### Kernel 4

![Kernel 4 Performance](plots/performance_kernel_4.png)

### Kernel 5

![Kernel 5 Performance](plots/performance_kernel_5.png)






## Stop and Remove TPU Instance

Do not forget to stop and remove the TPU instance.
```bash
$ ./remove_tpu_vm.sh
```


## License

MIT

## Acknowledgements

This project is inspired by JAX's Pallas framework and builds upon the TPU programming model.
