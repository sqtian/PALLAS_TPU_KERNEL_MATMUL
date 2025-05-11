"""
Utility functions for the TPU MatMul kernels project.
"""

from src.utils.benchmark import (
    matmul_flops,
    matmul_membw,
    matmul_flops_intensity,
    benchmark,
    get_matmul_performance,
    analyze_matmul,
    v5e_flops,
    v5e_membw,
    v5e_op_intensity
)

from src.utils.slice_utils import slices_for_invocation

__all__ = [
    'matmul_flops',
    'matmul_membw',
    'matmul_flops_intensity',
    'benchmark',
    'get_matmul_performance',
    'analyze_matmul',
    'slices_for_invocation',
    'v5e_flops',
    'v5e_membw',
    'v5e_op_intensity'
]
