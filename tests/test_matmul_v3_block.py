"""
Test for Matrix Multiplication Kernel V3 (Block-based)
"""

import unittest
import jax
import jax.numpy as jnp
from src.kernels.matmul_v3_block import run_matmul_v3

class TestMatmulV3Block(unittest.TestCase):
    def setUp(self):
        # Create test matrices
        self.key1, self.key2 = jax.random.split(jax.random.key(0))
        # Block sizes
        self.bm, self.bk, self.bn = 128, 128, 128
        # Create matrices with dimensions that are multiples of block sizes
        self.a_small = jax.random.normal(self.key1, (2*self.bm, 2*self.bk), dtype=jnp.float32)
        self.b_small = jax.random.normal(self.key2, (2*self.bk, 2*self.bn), dtype=jnp.float32)
        self.a_large = jax.random.normal(self.key1, (4*self.bm, 4*self.bk), dtype=jnp.float32)
        self.b_large = jax.random.normal(self.key2, (4*self.bk, 4*self.bn), dtype=jnp.float32)
    
    def test_small_matrix_default_blocks(self):
        # Test with small matrices using default block sizes
        result = run_matmul_v3(self.a_small, self.b_small)
        reference = jnp.matmul(self.a_small, self.b_small)
        self.assertTrue(jnp.allclose(result, reference))
    
    def test_small_matrix_custom_blocks(self):
        # Test with small matrices using custom block sizes
        bm, bk, bn = 64, 64, 64
        a = jax.random.normal(self.key1, (4*bm, 4*bk), dtype=jnp.float32)
        b = jax.random.normal(self.key2, (4*bk, 4*bn), dtype=jnp.float32)
        result = run_matmul_v3(a, b, bm=bm, bk=bk, bn=bn)
        reference = jnp.matmul(a, b)
        self.assertTrue(jnp.allclose(result, reference))
    
    def test_large_matrix(self):
        # Test with large matrices
        result = run_matmul_v3(self.a_large, self.b_large)
        reference = jnp.matmul(self.a_large, self.b_large)
        self.assertTrue(jnp.allclose(result, reference))
    
    def test_non_square_matrix(self):
        # Test with non-square matrices
        bm, bk, bn = 64, 64, 64
        a = jax.random.normal(self.key1, (4*bm, 6*bk), dtype=jnp.float32)
        b = jax.random.normal(self.key2, (6*bk, 2*bn), dtype=jnp.float32)
        result = run_matmul_v3(a, b, bm=bm, bk=bk, bn=bn)
        reference = jnp.matmul(a, b)
        self.assertTrue(jnp.allclose(result, reference))
        self.assertEqual(result.shape, (4*bm, 2*bn))

if __name__ == "__main__":
    unittest.main() 