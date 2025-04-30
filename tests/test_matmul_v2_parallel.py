"""
Test for Matrix Multiplication Kernel V2 (Parallel)
"""

import unittest
import jax
import jax.numpy as jnp
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.kernels.matmul_v2_parallel import run_matmul_v2

class TestMatmulV2Parallel(unittest.TestCase):
    def setUp(self):
        # Create test matrices
        self.key1, self.key2 = jax.random.split(jax.random.key(0))
        self.a_small = jax.random.normal(self.key1, (128, 128), dtype=jnp.float32)
        self.b_small = jax.random.normal(self.key2, (128, 128), dtype=jnp.float32)
        self.a_large = jax.random.normal(self.key1, (1024, 1024), dtype=jnp.float32)
        self.b_large = jax.random.normal(self.key2, (1024, 1024), dtype=jnp.float32)
    
    def test_small_matrix_n2(self):
        # Test with small matrices, N=2
        N = 2
        result = run_matmul_v2(self.a_small, self.b_small, N)
        reference = jnp.matmul(self.a_small, self.b_small)
        self.assertTrue(jnp.allclose(result, reference))
    
    def test_small_matrix_n4(self):
        # Test with small matrices, N=4
        N = 4
        result = run_matmul_v2(self.a_small, self.b_small, N)
        reference = jnp.matmul(self.a_small, self.b_small)
        self.assertTrue(jnp.allclose(result, reference))
    
    def test_large_matrix_n8(self):
        # Test with large matrices, N=8
        N = 8
        result = run_matmul_v2(self.a_large, self.b_large, N)
        reference = jnp.matmul(self.a_large, self.b_large)
        self.assertTrue(jnp.allclose(result, reference))
    
    def test_large_matrix_n16(self):
        # Test with large matrices, N=16
        N = 16
        result = run_matmul_v2(self.a_large, self.b_large, N)
        reference = jnp.matmul(self.a_large, self.b_large)
        self.assertTrue(jnp.allclose(result, reference))
    
    def test_non_square_matrix(self):
        # Test with non-square matrices
        N = 4
        a = jax.random.normal(self.key1, (256, 512), dtype=jnp.float32)
        b = jax.random.normal(self.key2, (512, 128), dtype=jnp.float32)
        result = run_matmul_v2(a, b, N)
        reference = jnp.matmul(a, b)
        self.assertTrue(jnp.allclose(result, reference))
        self.assertEqual(result.shape, (256, 128))

if __name__ == "__main__":
    unittest.main() 