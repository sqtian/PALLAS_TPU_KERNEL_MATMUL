"""
Matrix Multiplication Kernel V5 (Quantized Precision)

This module implements a BFloat16 matrix multiplication kernel using Pallas on TPU.
The kernel uses BFloat16 for computation but accumulates in Float32 for better precision.
"""

import jax
import jax.numpy as jnp
import functools
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def matmul_v5_quant_prec_kernel(a_ref, b_ref, o_ref, accu_ref, *, k_steps):
  """
  BFloat16 matrix multiplication kernel with float32 accumulation.

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
  accu_ref[...] += jnp.dot(a_ref[...], b_ref[...],
                           preferred_element_type=jnp.float32)

  @pl.when(pl.program_id(2) == k_steps - 1)
  def update_result():
    o_ref[...] = accu_ref[...].astype(o_ref.dtype)


@functools.partial(jax.jit, static_argnames=['bm', 'bk', 'bn'])
def run_matmul_v5(
    a: jax.Array,
    b: jax.Array,
    *,
    bm: int = 128,
    bk: int = 128,
        bn: int = 128):
  """
  Run the BFloat16 matrix multiplication kernel.

  Args:
      a: First input matrix of shape (m, k) in bfloat16
      b: Second input matrix of shape (k, n) in bfloat16
      bm: Block size for the m dimension
      bk: Block size for the k dimension
      bn: Block size for the n dimension

  Returns:
      Matrix product of a and b of shape (m, n) in bfloat16
  """
  m, k = a.shape
  _, n = b.shape
  assert k == b.shape[0]

  run_kernel = pl.pallas_call(
      functools.partial(matmul_v5_quant_prec_kernel, k_steps=k // bk),
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
  import sys
  import os
  sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))
  from src.utils.benchmark import analyze_matmul

  dtype = jnp.bfloat16

  # Test for correctness
  m, k, n = 4096, 4096, 4096
  bm, bk, bn = 128, 128, 128

  k1, k2 = jax.random.split(jax.random.key(0), 2)
  a = jax.random.normal(k1, (m, k), dtype=dtype)
  b = jax.random.normal(k2, (k, n), dtype=dtype)

  # Verify correctness - allow some tolerance due to bfloat16 precision
  result = run_matmul_v5(a, b, bm=bm, bk=bk, bn=bn)
  reference = jnp.matmul(a, b)
  is_correct = jnp.allclose(result, reference, atol=2)
  print(f"Correctness check: {'Success' if is_correct else 'Fail'}")

  # Benchmark
  analyze_matmul(m=m, k=k, n=n, dtype=dtype, mm_func=run_matmul_v5,
                 bm=bm, bk=bk, bn=bn)
