import jax
import jax.numpy as jnp
import numpy as np
import timeit
import datetime
import math

def matmul_flops(m: int, k: int, n: int):
    """Calculate the number of floating-point operations for a matrix multiplication."""
    return 2 * m * k * n

def matmul_membw(m: int, k: int, n: int, dtype: jnp.dtype):
    """Calculate the memory bandwidth required for a matrix multiplication."""
    return (m * k + k * n + m * n) * np.dtype(dtype).itemsize

def matmul_flops_intensity(m: int, k: int, n: int, dtype: jnp.dtype):
    """Calculate the flops intensity (flops per byte) for a matrix multiplication."""
    flops = matmul_flops(m, k, n)
    membw = matmul_membw(m, k, n, dtype)
    return flops / membw

# TPU v5e specs
v5e_flops = 197e12
v5e_membw = 819e9
v5e_op_intensity = v5e_flops / v5e_membw  # ~240.5

def benchmark(f, ntrials: int = 100):
    """Benchmark a function by running it multiple times and measuring the average execution time."""
    def run(*args, **kwargs):
        # Compile function first
        jax.block_until_ready(f(*args, **kwargs))
        # Time function
        result = timeit.timeit(lambda: jax.block_until_ready(f(*args, **kwargs)),
                           number=ntrials)
        time = result / ntrials
        return time
    return run

def get_matmul_performance(m: int, k: int, n: int, dtype: np.dtype,
                   mm_func, **kwargs):
    """Get the performance of a matrix multiplication kernel in FLOP/s."""
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    a = jax.random.normal(k1, (m, k), dtype=dtype)
    b = jax.random.normal(k2, (k, n), dtype=dtype)
    time = benchmark(mm_func)(a, b, **kwargs)
    mm_flops = matmul_flops(m, k, n) / time
    return mm_flops, time

def analyze_matmul(m: int, k: int, n: int, dtype: np.dtype,
                   mm_func, **kwargs):
    """Analyze the performance of a matrix multiplication kernel."""
    custom_flops, time = get_matmul_performance(m, k, n, dtype, mm_func, **kwargs)
    print(f"----- {m} x {k} x {n} -----")
    print("Matmul time: ", time)
    print("Matmul GFLOP/s: ", custom_flops / 10**9)
    print(f"FLOP/s utilization: {custom_flops / v5e_flops * 100:.4f}%")
    xla_flops, _ = get_matmul_performance(m, k, n, dtype, jnp.matmul)
    print(f"Percentage of XLA FLOP/s: {custom_flops / xla_flops * 100:.4f}%") 