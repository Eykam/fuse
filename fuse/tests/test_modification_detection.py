"""
Test script to verify variable modification detection in fusion functions.
"""

import pytest
import inspect
from fuse.ast_parser import KernelExtractor
from fuse.kernels import ops


def valid_fusion(a, b, c):
    """Valid fusion function with only kernel operations."""
    temp1 = ops.add(a, b)
    temp2 = ops.relu(temp1)
    result = ops.multiply(temp2, c)
    return result


def invalid_binary(a, b):
    """Invalid function with binary operation."""
    temp = ops.add(a, b)
    result = temp * 2  # Invalid binary operation
    return result


def invalid_aug_assign(a, b):
    """Invalid function with augmented assignment."""
    temp = ops.add(a, b)
    temp += 1  # Invalid augmented assignment
    result = ops.relu(temp)
    return result


def some_other_function(a):
    pass


def invalid_call(a, b):
    """Invalid function with non-kernel function call."""
    temp = ops.add(a, b)
    result = some_other_function(temp)  # Invalid function call
    return result


def valid_reassign(a, b, c):
    """Valid function with variable reassignment."""
    temp1 = ops.add(a, b)
    temp2 = temp1  # Valid variable assignment
    result = ops.multiply(temp2, c)
    return result


def test_valid_fusion():
    """Test that valid fusion function passes validation"""
    source = inspect.getsource(valid_fusion)
    extractor = KernelExtractor("valid_fusion")
    config = extractor.extract_from_source(source)
    assert len(config.kernel_calls) == 3


def test_valid_reassign():
    """Test that valid variable reassignment passes validation"""
    source = inspect.getsource(valid_reassign)
    extractor = KernelExtractor("valid_reassign")
    config = extractor.extract_from_source(source)
    assert len(config.kernel_calls) == 2


def test_invalid_binary():
    """Test that binary operations are rejected"""
    source = inspect.getsource(invalid_binary)
    extractor = KernelExtractor("invalid_binary")
    with pytest.raises(RuntimeError):
        extractor.extract_from_source(source)


def test_invalid_aug_assign():
    """Test that augmented assignments are rejected"""
    source = inspect.getsource(invalid_aug_assign)
    extractor = KernelExtractor("invalid_aug_assign")
    with pytest.raises(RuntimeError):
        extractor.extract_from_source(source)


def test_invalid_call():
    """Test that non-kernel function calls are rejected"""
    source = inspect.getsource(invalid_call)
    extractor = KernelExtractor("invalid_call")
    with pytest.raises(RuntimeError):
        extractor.extract_from_source(source)
