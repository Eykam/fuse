"""
Test class method detection in @fuse decorator
"""

import pytest
from fuse.fusion_decorator import fuse
from fuse.kernels import ops


# Truly module-level function
@fuse
def valid_module_function(a, b):
    return ops.add(a, b)


def test_module_level_function():
    """Test that module-level functions work"""
    # Test the module-level function (already defined above)
    assert hasattr(valid_module_function, '__qualname__')
    # Should not raise an exception, just checking it's decorated properly
    assert callable(valid_module_function)


def test_class_method_rejection():
    """Test that class methods are rejected"""
    with pytest.raises(RuntimeError, match="class methods or nested functions"):
        class TestClass:
            @fuse
            def invalid_method(self, a, b):
                return ops.add(a, b)


def test_nested_function_rejection():
    """Test that nested functions are rejected"""
    with pytest.raises(RuntimeError, match="class methods or nested functions"):
        def outer_function():
            @fuse
            def invalid_nested(a, b):
                return ops.add(a, b)

            return invalid_nested

        outer_function()