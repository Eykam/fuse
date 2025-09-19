"""
Kernel operations class for CUDA codegen.
Using a class makes it easier to identify kernel calls in the AST.
"""

import numpy as np


class Kernels:
    """Container for all kernel operations"""

    @staticmethod
    def add(a, b):
        """Element-wise addition"""
        return a + b

    @staticmethod
    def subtract(a, b):
        """Element-wise subtraction"""
        return a - b

    @staticmethod
    def multiply(a, b):
        """Element-wise multiplication"""
        return a * b

    @staticmethod
    def divide(a, b):
        """Element-wise division"""
        return a / b

    @staticmethod
    def relu(x):
        """ReLU activation: max(0, x)"""
        return np.maximum(0, x)


# Create a global instance for convenience
ops = Kernels()