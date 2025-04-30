"""
Test for Matrix Multiplication Kernel V6 (INT8)
"""

import unittest
import jax
import jax.numpy as jnp
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.kernels.matmul_v6_int8 import run_matmul_v6

class TestMatmulV6Int8(unittest.TestCase):
    def setUp(self):
        # Create test matrices
        self.key1, self.key2 = jax.random.split(jax.random.key(0))
        # Block sizes
        self.bm, self.bk, self.bn = 512, 1024, 1024
        # For now, we're using bfloat16 for the tests since the kernel currently uses that
        # In a real INT8 implementation, we would use int8 dtype
        self.dtype = jnp.bfloat16
        
        # Create matrices with dimensions that are multiples of block sizes
        self.a_small = jax.random.normal(self.key1, (self.bm, self.bk), dtype=self.dtype)
        self.b_small = jax.random.normal(self.key2, (self.bk, self.bn), dtype=self.dtype)
        self.a_large = jax.random.normal(self.key1, (2*self.bm, 2*self.bk), dtype=self.dtype)
        self.b_large = jax.random.normal(self.key2, (2*self.bk, 2*self.bn), dtype=self.dtype)
    
    def test_small_matrix_default_blocks(self):
        # Test with small matrices using default block sizes
        result = run_matmul_v6(self.a_small, self.b_small)
        reference = jnp.matmul(self.a_small, self.b_small)
        # Use atol=2 because of bfloat16 precision
        self.assertTrue(jnp.allclose(result, reference, atol=2))
    
    def test_small_matrix_custom_blocks(self):
        # Test with small matrices using custom block sizes
        bm, bk, bn = 256, 512, 512
        a = jax.random.normal(self.key1, (2*bm, 2*bk), dtype=self.dtype)
        b = jax.random.normal(self.key2, (2*bk, 2*bn), dtype=self.dtype)
        result = run_matmul_v6(a, b, bm=bm, bk=bk, bn=bn)
        reference = jnp.matmul(a, b)
        self.assertTrue(jnp.allclose(result, reference, atol=2))
    
    def test_large_matrix(self):
        # Test with large matrices
        result = run_matmul_v6(self.a_large, self.b_large)
        reference = jnp.matmul(self.a_large, self.b_large)
        self.assertTrue(jnp.allclose(result, reference, atol=2))
    
    def test_ones_matrix(self):
        # Test with matrices filled with ones
        a = jnp.ones((2*self.bm, 2*self.bk), dtype=self.dtype)
        b = jnp.ones((2*self.bk, 2*self.bn), dtype=self.dtype)
        result = run_matmul_v6(a, b)
        reference = jnp.matmul(a, b)
        self.assertTrue(jnp.allclose(result, reference, atol=2))
    
    def test_very_large_matrix(self):
        # Test with very large matrices - this should be run only when sufficient memory is available
        # Skip by default
        self.skipTest("Skipping very large matrix test to avoid memory issues")
        
        m, k, n = 8192, 8192, 8192
        a = jnp.ones((m, k), dtype=self.dtype)
        b = jnp.ones((k, n), dtype=self.dtype)
        result = run_matmul_v6(a, b)
        reference = jnp.matmul(a, b)
        self.assertTrue(jnp.allclose(result, reference, atol=2))

if __name__ == "__main__":
    unittest.main() 