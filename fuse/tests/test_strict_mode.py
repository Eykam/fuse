"""Test kernel fusion with strict mode enabled."""

import numpy as np
import pytest
from fuse.fusion_decorator import fuse
from fuse.kernels import ops


@fuse(strict=True)
def simple_add(a, b):
    """Simple addition kernel."""
    return ops.add(a, b)


def test_strict_mode_execution():
    """Test that kernel executes successfully with strict mode."""
    # Create input arrays
    a = np.array([1, 2, 3, 4], dtype=np.float32)
    b = np.array([5, 6, 7, 8], dtype=np.float32)

    # Execute fused kernel - will raise exception if compilation/execution fails
    result = simple_add(a, b)

    # Verify result
    expected = np.array([6, 8, 10, 12], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)

    print("Strict mode test passed!")


if __name__ == "__main__":
    test_strict_mode_execution()
