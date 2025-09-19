"""
Test the integrated FFI in the fusion decorator
"""

import pytest
import numpy as np
from fuse.fusion_decorator import fuse
from fuse.kernels import ops


@fuse
def simple_add_relu(a, b):
    temp = ops.add(a, b)
    result = ops.relu(temp)
    return result


@pytest.fixture
def test_arrays_1024():
    """Create test arrays with shape 1024"""
    a = np.random.randn(1024).astype(np.float32)
    b = np.random.randn(1024).astype(np.float32)
    return a, b


@pytest.fixture
def test_arrays_2048():
    """Create test arrays with shape 2048"""
    c = np.random.randn(2048).astype(np.float32)
    d = np.random.randn(2048).astype(np.float32)
    return c, d


def test_ffi_integration_first_call(test_arrays_1024):
    """Test first call (cache miss)"""
    a, b = test_arrays_1024
    result = simple_add_relu(a, b)
    assert result.shape == (1024,)
    assert result.dtype == np.float32


def test_ffi_integration_second_call(test_arrays_1024):
    """Test second call with same shape (cache hit)"""
    a, b = test_arrays_1024
    result1 = simple_add_relu(a, b)
    result2 = simple_add_relu(a, b)
    assert result1.shape == result2.shape
    assert result1.dtype == result2.dtype == np.float32


def test_ffi_integration_different_shape(test_arrays_2048):
    """Test call with different shape (cache miss)"""
    c, d = test_arrays_2048
    result = simple_add_relu(c, d)
    assert result.shape == (2048,)
    assert result.dtype == np.float32
