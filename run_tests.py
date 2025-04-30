#!/usr/bin/env python3
"""
Helper script to run tests for the TPU matmul kernels project.
This script ensures the src package is in the Python path.
"""

import os
import sys
import unittest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests')
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_runner.run(test_suite) 