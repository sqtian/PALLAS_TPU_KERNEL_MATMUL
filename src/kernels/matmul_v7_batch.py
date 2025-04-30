"""
Matrix Multiplication Kernel V7 (Batch with Activation)

This module implements a parallel matrix multiplication kernel with activation function
support for processing batched inputs.
"""

import jax
import jax.numpy as jnp
import functools
from jax.experimental import pallas as pl
from typing import Callable, Optional

def matmul_v7_batch_kernel(a_ref, b_ref, o_ref, *, activation):
    """
    Batch matrix multiplication kernel with optional activation function.
    
    Args:
        a_ref: Reference to the first input matrix slice
        b_ref: Reference to the second input matrix slice
        o_ref: Reference to the output matrix slice
        activation: Activation function to apply to the matrix product
    """
    o_ref[...] = activation(a_ref[...] @ b_ref[...])

@functools.partial(jax.jit, static_argnames=['N', 'activation'])
def run_matmul_v7(
    a: jax.Array, 
    b: jax.Array, 
    N: int, 
    activation: Callable = lambda x: x):
    """
    Run the batch matrix multiplication kernel with activation function.
    
    Args:
        a: First input matrix of shape (m, k)
        b: Second input matrix of shape (k, n)
        N: Number of partitions in each dimension (grid size)
        activation: Activation function to apply to the result (default: identity)
        
    Returns:
        Matrix product of a and b with activation applied, of shape (m, n)
    """
    kernel = pl.pallas_call(
        functools.partial(matmul_v7_batch_kernel, activation=activation),
        grid=(N, N),
        in_specs=[
            pl.BlockSpec((a.shape[0] // N, a.shape[1]), lambda i, j: (i, 0)),
            pl.BlockSpec((b.shape[0], b.shape[1] // N), lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec((a.shape[0] // N, b.shape[1] // N), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), a.dtype)
    )
    return kernel(a, b)

def batch_matmul_with_activation(
    a: jax.Array, 
    b: jax.Array, 
    N: int = 8, 
    activation: Callable = jax.nn.relu):
    """
    Perform batched matrix multiplication with activation function.
    
    Args:
        a: Batched input tensor of shape (batch_size, m, k)
        b: Batched input tensor of shape (batch_size, k, n)
        N: Number of partitions in each dimension (grid size)
        activation: Activation function to apply to the result
        
    Returns:
        Batched matrix product with activation applied, of shape (batch_size, m, n)
    """
    return jax.vmap(
        functools.partial(run_matmul_v7, N=N, activation=activation), 
        in_axes=(0, 0)
    )(a, b)

if __name__ == "__main__":
    # Simple test
    import datetime
    import math
    
    # Test for correctness
    k1, k2 = jax.random.split(jax.random.key(0))
    a = jax.random.normal(k1, (4, 1024, 1024), dtype=jnp.float32)
    b = jax.random.normal(k2, (4, 1024, 1024), dtype=jnp.float32)
    N = 8
    
    # Time the execution
    start_time = datetime.datetime.now()
    result = batch_matmul_with_activation(a, b, N=N)
    jax.block_until_ready(result)
    end_time = datetime.datetime.now()
    
    # Calculate performance
    elapsed_seconds = (end_time - start_time).total_seconds()
    flops = 2 * math.prod(a.shape) * b.shape[-1]
    print(f'Time taken: {end_time - start_time}')
    print(f'Performance: {flops / elapsed_seconds / 10**9} GFLOP/s')
    
    # Verify correctness
    reference = jax.nn.relu(jax.vmap(jnp.matmul)(a, b))
    is_correct = jnp.allclose(result, reference)
    print(f"Correctness check: {'Success' if is_correct else 'Fail'}") 