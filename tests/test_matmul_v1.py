"""
Test for Matrix Multiplication Kernel V1 (Naive)
"""

import unittest
import jax
import jax.numpy as jnp
import os
import sys
# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.kernels.matmul_v1 import run_matmul_v1

class TestMatmulV1(unittest.TestCase):
    def setUp(self):
        # Create test matrices
        self.key1, self.key2 = jax.random.split(jax.random.key(0))
        self.a_small = jax.random.normal(self.key1, (128, 128), dtype=jnp.float32)
        self.b_small = jax.random.normal(self.key2, (128, 128), dtype=jnp.float32)
        self.a_large = jax.random.normal(self.key1, (1024, 1024), dtype=jnp.float32)
        self.b_large = jax.random.normal(self.key2, (1024, 1024), dtype=jnp.float32)
    
    def test_small_matrix(self):
        # Test with small matrices
        result = run_matmul_v1(self.a_small, self.b_small)
        reference = jnp.matmul(self.a_small, self.b_small)
        self.assertTrue(jnp.allclose(result, reference))
    
    def test_large_matrix(self):
        # Test with large matrices
        result = run_matmul_v1(self.a_large, self.b_large)
        reference = jnp.matmul(self.a_large, self.b_large)
        self.assertTrue(jnp.allclose(result, reference))
    
    def test_non_square_matrix(self):
        # Test with non-square matrices
        a = jax.random.normal(self.key1, (256, 512), dtype=jnp.float32)
        b = jax.random.normal(self.key2, (512, 128), dtype=jnp.float32)
        result = run_matmul_v1(a, b)
        reference = jnp.matmul(a, b)
        self.assertTrue(jnp.allclose(result, reference))
        self.assertEqual(result.shape, (256, 128))

if __name__ == "__main__":
    unittest.main() 