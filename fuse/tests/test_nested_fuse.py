"""
Test nested @fuse function call detection
"""

import pytest
from fuse.ast_parser import KernelExtractor
from fuse.kernels import ops
from fuse.fusion_decorator import fuse


@fuse
def inner_func(a, b):
    """Inner @fuse function"""
    return ops.add(a, b)


@fuse
def outer_func(x, y, z):
    """Outer @fuse function that calls inner @fuse function"""
    temp = inner_func(x, y)  # This should trigger an error
    result = ops.relu(temp)
    return result


def test_nested_fuse_detection():
    """Test that nested @fuse calls are detected and rejected"""
    # Create source with both function definitions
    source = """
from kernels import ops
from fusion_decorator import fuse

@fuse
def inner_func(a, b):
    '''Inner @fuse function'''
    return ops.add(a, b)

@fuse
def outer_func(x, y, z):
    '''Outer @fuse function that calls inner @fuse function'''
    temp = inner_func(x, y)  # This should trigger an error
    result = ops.relu(temp)
    return result
"""

    with pytest.raises(RuntimeError, match="Nested @fuse function call"):
        extractor = KernelExtractor("outer_func")
        extractor.extract_from_source(source)
