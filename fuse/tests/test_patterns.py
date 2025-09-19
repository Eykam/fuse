"""
Example patterns demonstrating different kernel fusion scenarios.
"""

import pytest
import numpy as np
from fuse.kernels import ops
from fuse.fusion_decorator import fuse


@fuse
def pattern1_simple_chain(a, b):
    """Simple chain: add -> relu"""
    temp = ops.add(a, b)
    result = ops.relu(temp)
    return result


@fuse
def pattern1_longer_chain(x, y, z):
    """Longer chain: multiply -> add -> relu"""
    temp1 = ops.multiply(x, y)
    temp2 = ops.add(temp1, z)
    result = ops.relu(temp2)
    return result


@fuse
def pattern2_parallel(x, y, z):
    """Parallel operations on different data"""
    result1 = ops.relu(x)
    result2 = ops.multiply(y, z)
    return result1, result2


@fuse
def pattern3_fork_join(x):
    """Fork-join: diverge then converge"""
    branch1 = ops.relu(x)
    branch2 = ops.multiply(x, x)
    result = ops.add(branch1, branch2)
    return result


@fuse
def pattern4_complex(a, b, c):
    """Complex pattern with nested operations and reuse"""
    # Nested operation
    temp1 = ops.add(ops.multiply(a, b), c)
    temp2 = ops.relu(temp1)
    # Reuse input 'a'
    result = ops.subtract(temp2, a)
    return result


@fuse
def pattern5_multi_fork_join(x, y, z):
    """Multiple forks and joins"""
    # Fork from x
    fork1 = ops.relu(ops.add(x, y))
    fork2 = ops.multiply(x, z)
    fork3 = ops.subtract(x, ops.multiply(y, z))

    # Partial joins
    join1 = ops.add(fork1, fork2)
    join2 = ops.multiply(fork2, fork3)

    # Final join
    result = ops.relu(ops.add(join1, join2))
    return result


@pytest.fixture
def test_data():
    """Fixture to provide test data arrays."""
    a = np.random.randn(1024).astype(np.float32)
    b = np.random.randn(1024).astype(np.float32)
    c = np.random.randn(1024).astype(np.float32)
    return a, b, c


def test_pattern1_simple_chain(test_data):
    """Test simple chain: add -> relu"""
    a, b, _ = test_data
    result = pattern1_simple_chain(a, b)
    assert result.shape == a.shape
    assert result.dtype == np.float32


def test_pattern1_longer_chain(test_data):
    """Test longer chain: multiply -> add -> relu"""
    a, b, c = test_data
    result = pattern1_longer_chain(a, b, c)
    assert result.shape == a.shape
    assert result.dtype == np.float32


def test_pattern2_parallel(test_data):
    """Test parallel operations on different data"""
    a, b, c = test_data
    result1, result2 = pattern2_parallel(a, b, c)
    assert result1.shape == a.shape
    assert result2.shape == b.shape
    assert result1.dtype == np.float32
    assert result2.dtype == np.float32


def test_pattern3_fork_join(test_data):
    """Test fork-join: diverge then converge"""
    a, _, _ = test_data
    result = pattern3_fork_join(a)
    assert result.shape == a.shape
    assert result.dtype == np.float32


def test_pattern4_complex(test_data):
    """Test complex pattern with nested operations and reuse"""
    a, b, c = test_data
    result = pattern4_complex(a, b, c)
    assert result.shape == a.shape
    assert result.dtype == np.float32


def test_pattern5_multi_fork_join(test_data):
    """Test multiple forks and joins"""
    a, b, c = test_data
    result = pattern5_multi_fork_join(a, b, c)
    assert result.shape == a.shape
    assert result.dtype == np.float32
