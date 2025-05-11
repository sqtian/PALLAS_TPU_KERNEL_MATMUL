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


def matmul_v4_block_kernel(a_ref, b_ref, o_ref):
  """
  Block-based matrix multiplication kernel.

  Args:
      a_ref: Reference to the first input matrix block
      b_ref: Reference to the second input matrix block
      o_ref: Reference to the output matrix block
  """
  @pl.when(pl.program_id(2) == 0)
  def init():
    o_ref[...] = jnp.zeros_like(o_ref)
  # Accumulates the multiplication for this block.
  o_ref[...] += a_ref[...] @ b_ref[...]


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
  m, k = a.shape
  _, n = b.shape
  assert k == b.shape[0]

  run_kernel = pl.pallas_call(
      matmul_v3_block_kernel,
      grid=(m // bm, n // bn, k // bk),
      in_specs=[
          pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
          pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
      ],
      out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
      out_shape=jax.ShapeDtypeStruct((m, n), a.dtype),
  )
  return run_kernel(a, b)


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
