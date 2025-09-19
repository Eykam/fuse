import numpy as np
from fuse.fusion_decorator import fuse


@fuse
def passthrough(a):
    return a


def test_passthrough():
    """Test passthrough kernel (identity function)"""
    a = np.ones(1024, dtype=np.float32)
    result = passthrough(a)

    assert result.shape == a.shape
    assert result.dtype == a.dtype
    assert np.array_equal(result, a)
