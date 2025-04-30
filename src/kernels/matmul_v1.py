"""
Matrix Multiplication Kernel V1 (Naive)

This module implements a naive matrix multiplication kernel using Pallas on TPU.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

def matmul_v1_kernel(a_ref, b_ref, o_ref):
    """
    Naive matrix multiplication kernel that directly uses the @ operator.
    
    Args:
        a_ref: Reference to the first input matrix
        b_ref: Reference to the second input matrix
        o_ref: Reference to the output matrix
    """
    o_ref[...] = a_ref[...] @ b_ref[...]

@jax.jit
def run_matmul_v1(a: jax.Array, b: jax.Array):
    """
    Run the naive matrix multiplication kernel.
    
    Args:
        a: First input matrix of shape (m, k)
        b: Second input matrix of shape (k, n)
        
    Returns:
        Matrix product of a and b of shape (m, n)
    """
    kernel = pl.pallas_call(
        matmul_v1_kernel,
        out_shape=jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), a.dtype)
    )
    return kernel(a, b)

if __name__ == "__main__":
    # Simple test
    from src.utils.benchmark import analyze_matmul
    
    # Test for correctness
    k1, k2 = jax.random.split(jax.random.key(0))
    a = jax.random.normal(k1, (1024, 1024), dtype=jnp.float32)
    b = jax.random.normal(k2, (1024, 1024), dtype=jnp.float32)
    
    # Verify correctness
    result = run_matmul_v1(a, b)
    reference = jnp.matmul(a, b)
    is_correct = jnp.allclose(result, reference)
    print(f"Correctness check: {'Success' if is_correct else 'Fail'}")
    
    # Benchmark
    analyze_matmul(m=1024, k=1024, n=1024, dtype=jnp.float32, mm_func=run_matmul_v1) 