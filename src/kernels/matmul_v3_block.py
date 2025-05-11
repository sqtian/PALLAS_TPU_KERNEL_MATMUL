"""
Matrix Multiplication Kernel V3 (Block-based)

This module implements a block-based matrix multiplication kernel using Pallas on TPU.
The kernel uses a 3D grid to process blocks of the input matrices.
"""

import jax
import jax.numpy as jnp
import functools
from jax.experimental import pallas as pl


def matmul_v3_block_kernel(a_ref, b_ref, o_ref):
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
def run_matmul_v3(
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
  m, k, n = 4096, 4096, 4096
  bm, bk, bn = 128, 128, 128
  dtype = jnp.float32

  k1, k2 = jax.random.split(jax.random.key(0), 2)
  a = jax.random.normal(k1, (m, k), dtype=dtype)
  b = jax.random.normal(k2, (k, n), dtype=dtype)

  # Verify correctness
  result = run_matmul_v3(a, b, bm=bm, bk=bk, bn=bn)
  reference = jnp.matmul(a, b)
  is_correct = jnp.allclose(result, reference)
  print(f"Correctness check: {'Success' if is_correct else 'Fail'}")

  # Benchmark
  analyze_matmul(m=m, k=k, n=n, dtype=dtype, mm_func=run_matmul_v3,
                 bm=bm, bk=bk, bn=bn)
