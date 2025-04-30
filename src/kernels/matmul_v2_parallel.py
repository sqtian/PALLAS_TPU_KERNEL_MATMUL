"""
Matrix Multiplication Kernel V2 (Parallel)

This module implements a parallel matrix multiplication kernel using Pallas on TPU.
The kernel splits matrices into row and column blocks for parallel processing.
"""

import jax
import jax.numpy as jnp
import functools
from jax.experimental import pallas as pl
from typing import Callable

def matmul_v2_parallel_kernel(a_ref, b_ref, o_ref):
    """
    Parallel matrix multiplication kernel that processes blocks of the matrices.
    
    Args:
        a_ref: Reference to the first input matrix slice
        b_ref: Reference to the second input matrix slice
        o_ref: Reference to the output matrix slice
    """
    o_ref[...] = a_ref[...] @ b_ref[...]

@functools.partial(jax.jit, static_argnames=['N'])
def run_matmul_v2(a: jax.Array, b: jax.Array, N: int):
    """
    Run the parallel matrix multiplication kernel.
    
    Args:
        a: First input matrix of shape (m, k)
        b: Second input matrix of shape (k, n)
        N: Number of partitions in each dimension (grid size)
        
    Returns:
        Matrix product of a and b of shape (m, n)
    """
    kernel = pl.pallas_call(
        matmul_v2_parallel_kernel,
        grid=(N, N),
        in_specs=[
            pl.BlockSpec((a.shape[0] // N, a.shape[1]), lambda i, j: (i, 0)),
            pl.BlockSpec((b.shape[0], b.shape[1] // N), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((a.shape[0] // N, b.shape[1] // N), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), a.dtype)
    )
    return kernel(a, b)

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from src.utils.benchmark import analyze_matmul
    
    # Test for correctness
    k1, k2 = jax.random.split(jax.random.key(0))
    a = jax.random.normal(k1, (1024, 1024), dtype=jnp.float32)
    b = jax.random.normal(k2, (1024, 1024), dtype=jnp.float32)
    N = 8  # Number of partitions
    
    # Verify correctness
    result = run_matmul_v2(a, b, N)
    reference = jnp.matmul(a, b)
    is_correct = jnp.allclose(result, reference)
    print(f"Correctness check: {'Success' if is_correct else 'Fail'}")
    
    # Benchmark
    analyze_matmul(m=1024, k=1024, n=1024, dtype=jnp.float32, mm_func=run_matmul_v2, N=N) 