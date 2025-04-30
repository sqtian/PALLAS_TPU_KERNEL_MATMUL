"""
Test package for TPU MatMul kernels.
"""

from tests.test_matmul_v1 import TestMatmulV1
from tests.test_matmul_v2_parallel import TestMatmulV2Parallel
from tests.test_matmul_v3_block import TestMatmulV3Block
from tests.test_matmul_v4_bf16 import TestMatmulV4Bf16
from tests.test_matmul_v5_large_block import TestMatmulV5LargeBlock
from tests.test_matmul_v6_int8 import TestMatmulV6Int8
from tests.test_matmul_v7_batch import TestMatmulV7Batch

__all__ = [
    'TestMatmulV1',
    'TestMatmulV2Parallel',
    'TestMatmulV3Block',
    'TestMatmulV4Bf16',
    'TestMatmulV5LargeBlock',
    'TestMatmulV6Int8',
    'TestMatmulV7Batch',
] 