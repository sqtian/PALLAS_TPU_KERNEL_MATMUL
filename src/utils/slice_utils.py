import jax
from jax.experimental import pallas as pl


def slices_for_invocation(x_shape: tuple[int, ...],
                          x_spec: pl.BlockSpec,
                          grid: tuple[int, ...],
                          invocation_indices: tuple[int, ...]) -> tuple[slice, ...]:
  """
  Calculate the slices for a specific invocation of a Pallas kernel.

  Args:
      x_shape: The shape of the input tensor.
      x_spec: The BlockSpec defining how the tensor is sliced.
      grid: The grid dimensions for the kernel.
      invocation_indices: The indices of the current invocation.

  Returns:
      A tuple of slices representing the portion of the tensor to be processed.
  """
  assert len(invocation_indices) == len(grid)
  assert all(0 <= i < grid_size for i,
             grid_size in zip(invocation_indices, grid))
  block_indices = x_spec.index_map(*invocation_indices)
  assert len(x_shape) == len(x_spec.block_shape) == len(block_indices)
  elem_indices = []
  for x_size, block_size, block_idx in zip(x_shape, x_spec.block_shape, block_indices):
    start_idx = block_idx * block_size
    # At least one element of the block must be within bounds
    assert start_idx < x_size
    elem_indices.append(slice(start_idx, start_idx + block_size))
  return tuple(elem_indices)
