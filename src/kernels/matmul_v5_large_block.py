"""
Matrix Multiplication Kernel V5 (Large Block)

This module implements a matrix multiplication kernel using larger block sizes
for optimal memory access patterns on TPU.
"""

import jax
import jax.numpy as jnp
import functools
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def matmul_v5_large_block_kernel(a_ref, b_ref, o_ref, accu_ref, *, k_steps):
    """
    Large block matrix multiplication kernel with float32 accumulation.
    
    Args:
        a_ref: Reference to the first input matrix block
        b_ref: Reference to the second input matrix block
        o_ref: Reference to the output matrix block
        accu_ref: Reference to the accumulator for higher precision
        k_steps: Total number of k dimension steps
    """
    @pl.when(pl.program_id(2) == 0)
    def init():
        accu_ref[...] = jnp.zeros_like(accu_ref)
    
    # Accumulates in higher precision
    accu_ref[...] += jnp.dot(a_ref[...], b_ref[...], preferred_element_type=jnp.float32)
    
    @pl.when(pl.program_id(2) == k_steps-1)
    def update_result():
        o_ref[...] = accu_ref[...].astype(o_ref.dtype)

@functools.partial(jax.jit, static_argnames=['bm', 'bk', 'bn'])
def run_matmul_v5(
    a: jax.Array,
    b: jax.Array,
    *,
    bm: int = 512,
    bk: int = 1024,
    bn: int = 1024):
    """
    Run the large block matrix multiplication kernel.
    
    Args:
        a: First input matrix of shape (m, k)
        b: Second input matrix of shape (k, n)
        bm: Block size for the m dimension (default: 512)
        bk: Block size for the k dimension (default: 1024)
        bn: Block size for the n dimension (default: 1024)
        
    Returns:
        Matrix product of a and b of shape (m, n)
    """
    m, k = a.shape
    _, n = b.shape
    assert k == b.shape[0]
    
    run_kernel = pl.pallas_call(
        functools.partial(matmul_v5_large_block_kernel, k_steps=k // bk),
        grid=(m // bm, n // bn, k // bk),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
            pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
        ],
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
        out_shape=jax.ShapeDtypeStruct((m, n), a.dtype),
        compiler_params=pltpu.TPUCompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
    )
    return run_kernel(a, b)

if __name__ == "__main__":
    # Simple test
    from src.utils.benchmark import analyze_matmul
    
    # Test for correctness
    m, k, n = 8192, 8192, 8192
    bm, bk, bn = 512, 1024, 1024
    
    # Use bfloat16 for better performance
    dtype = jnp.bfloat16
    
    k1, k2 = jax.random.split(jax.random.key(0), 2)
    a = jnp.ones((m, k), dtype=dtype)
    b = jnp.ones((k, n), dtype=dtype)
    
    # Verify correctness - allow some tolerance due to bfloat16 precision
    result = run_matmul_v5(a, b, bm=bm, bk=bk, bn=bn)
    reference = jnp.matmul(a, b)
    is_correct = jnp.allclose(result, reference, atol=2)
    print(f"Correctness check: {'Success' if is_correct else 'Fail'}")
    
    # Benchmark
    analyze_matmul(m=m, k=k, n=n, dtype=dtype, mm_func=run_matmul_v5, 
                  bm=bm, bk=bk, bn=bn) 