"""
Test for Matrix Multiplication Kernel V7 (Batch with Activation)
"""

import unittest
import jax
import jax.numpy as jnp
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.kernels.matmul_v7_batch import run_matmul_v7, batch_matmul_with_activation

class TestMatmulV7Batch(unittest.TestCase):
    def setUp(self):
        # Create test matrices
        self.key1, self.key2 = jax.random.split(jax.random.key(0))
        # Non-batched matrices
        self.a = jax.random.normal(self.key1, (1024, 1024), dtype=jnp.float32)
        self.b = jax.random.normal(self.key2, (1024, 1024), dtype=jnp.float32)
        # Batched matrices
        self.batch_size = 4
        self.a_batch = jax.random.normal(self.key1, (self.batch_size, 1024, 1024), dtype=jnp.float32)
        self.b_batch = jax.random.normal(self.key2, (self.batch_size, 1024, 1024), dtype=jnp.float32)
    
    def test_single_matmul_no_activation(self):
        # Test single matrix multiplication without activation
        N = 8
        identity = lambda x: x
        result = run_matmul_v7(self.a, self.b, N, activation=identity)
        reference = jnp.matmul(self.a, self.b)
        self.assertTrue(jnp.allclose(result, reference))
    
    def test_single_matmul_with_relu(self):
        # Test single matrix multiplication with ReLU activation
        N = 8
        result = run_matmul_v7(self.a, self.b, N, activation=jax.nn.relu)
        reference = jax.nn.relu(jnp.matmul(self.a, self.b))
        self.assertTrue(jnp.allclose(result, reference))
    
    def test_batch_matmul_with_relu(self):
        # Test batched matrix multiplication with ReLU activation
        N = 8
        result = batch_matmul_with_activation(self.a_batch, self.b_batch, N=N, activation=jax.nn.relu)
        reference = jax.nn.relu(jax.vmap(jnp.matmul)(self.a_batch, self.b_batch))
        self.assertTrue(jnp.allclose(result, reference))
    
    def test_batch_matmul_with_tanh(self):
        # Test batched matrix multiplication with tanh activation
        N = 8
        result = batch_matmul_with_activation(self.a_batch, self.b_batch, N=N, activation=jnp.tanh)
        reference = jnp.tanh(jax.vmap(jnp.matmul)(self.a_batch, self.b_batch))
        self.assertTrue(jnp.allclose(result, reference))
    
    def test_different_grid_sizes(self):
        # Test with different grid sizes
        for N in [2, 4, 8, 16]:
            result = batch_matmul_with_activation(self.a_batch, self.b_batch, N=N)
            reference = jax.nn.relu(jax.vmap(jnp.matmul)(self.a_batch, self.b_batch))
            self.assertTrue(jnp.allclose(result, reference))

if __name__ == "__main__":
    unittest.main() 