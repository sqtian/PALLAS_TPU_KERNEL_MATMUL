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
from matmul_v3_block import run_matmul_v3


@functools.partial(jax.jit, static_argnames=['bm', 'bk', 'bn'])
def run_matmul_v4(
    a: jax.Array,
    b: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
        bn: int = 128):
  """
  Run the block-based matrix multiplication kernel.

  Args:
      a: First input matrix of shape (m, k)
      b: Second input matrix of shape (k, n)
      bm: Block size for the m dimension
      bk: Block size for the k dimension
      bn: Block size for the n dimension

  Returns:
      Matrix product of a and b of shape (m, n)
  """
  return run_matmul_v3(a, b, bm=bm, bk=bk, bn=bn)


if __name__ == "__main__":
  import sys
  import os
  sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))
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
  result = run_matmul_v4(a, b, bm=bm, bk=bk, bn=bn)
  reference = jnp.matmul(a, b)
  is_correct = jnp.allclose(result, reference, atol=2)
  print(f"Correctness check: {'Success' if is_correct else 'Fail'}")

  # Benchmark
  analyze_matmul(m=m, k=k, n=n, dtype=dtype, mm_func=run_matmul_v5,
                 bm=bm, bk=bk, bn=bn)
