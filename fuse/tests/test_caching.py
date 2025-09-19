"""Test kernel caching mechanism."""

import numpy as np
import pytest
from fuse.fusion_decorator import fuse, _fusion_cache, extract_tensor_info
from fuse.kernels import ops


@fuse(strict=True)
def cached_add(a, b):
    """Addition kernel for testing caching."""
    return ops.add(a, b)


def get_kernel_id_for_function(func, *args):
    """Extract the kernel_id that would be used for this function call."""
    # Simulate the cache key generation logic from the decorator
    import inspect

    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    # Extract tensor info from runtime arguments
    tensor_infos = {}
    for i, (param_name, arg) in enumerate(zip(param_names, args)):
        tensor_infos[param_name] = extract_tensor_info(arg)

    # Generate cache key
    cache_key = _fusion_cache.get_cache_key(func.__name__, tensor_infos)

    # Check if it's in cache
    cache_entry = _fusion_cache.get(cache_key)
    return cache_entry.kernel_id if cache_entry else None


def test_caching_mechanism():
    """Test that successive calls with same shape but different values reuse cached kernels."""
    # All arrays have same shape (4,) but different values
    print("=== Testing caching with same shape, different values ===")

    # First set of inputs
    a1 = np.array([1, 2, 3, 4], dtype=np.float32)
    b1 = np.array([5, 6, 7, 8], dtype=np.float32)
    expected1 = np.array([6, 8, 10, 12], dtype=np.float32)

    # Second set of inputs (same shape, different values)
    a2 = np.array([10, 20, 30, 40], dtype=np.float32)
    b2 = np.array([1, 2, 3, 4], dtype=np.float32)
    expected2 = np.array([11, 22, 33, 44], dtype=np.float32)

    # Third set of inputs (same shape, different values)
    a3 = np.array([100, 200, 300, 400], dtype=np.float32)
    b3 = np.array([50, 60, 70, 80], dtype=np.float32)
    expected3 = np.array([150, 260, 370, 480], dtype=np.float32)

    print("Before any calls - checking cache:")
    kernel_id_before = get_kernel_id_for_function(cached_add, a1, b1)
    print(f"Kernel ID in cache: {kernel_id_before}")
    assert kernel_id_before is None, "Cache should be empty initially"

    print("\n=== First call with [1,2,3,4] + [5,6,7,8] (should compile) ===")
    result1 = cached_add(a1, b1)
    np.testing.assert_array_equal(result1, expected1)

    kernel_id_after_first = get_kernel_id_for_function(cached_add, a1, b1)
    print(f"Kernel ID after first call: {kernel_id_after_first}")
    assert kernel_id_after_first is not None, "Kernel should be cached after first call"

    print("\n=== Second call with [10,20,30,40] + [1,2,3,4] (should reuse cache - same shape!) ===")
    result2 = cached_add(a2, b2)
    np.testing.assert_array_equal(result2, expected2)

    kernel_id_after_second = get_kernel_id_for_function(cached_add, a2, b2)
    print(f"Kernel ID after second call: {kernel_id_after_second}")
    assert kernel_id_after_second == kernel_id_after_first, "Same shape inputs should reuse cached kernel"

    print("\n=== Third call with [100,200,300,400] + [50,60,70,80] (should reuse cache - same shape!) ===")
    result3 = cached_add(a3, b3)
    np.testing.assert_array_equal(result3, expected3)

    kernel_id_after_third = get_kernel_id_for_function(cached_add, a3, b3)
    print(f"Kernel ID after third call: {kernel_id_after_third}")
    assert kernel_id_after_third == kernel_id_after_first, "Same shape inputs should still reuse cached kernel"

    print(f"\n✅ Caching test passed! All same-shape calls used kernel ID: {kernel_id_after_first}")
    print("✅ Kernel was compiled once and reused for all same-shape inputs regardless of values!")


def test_different_shapes_different_cache():
    """Test that different input shapes create different cache entries."""
    print("\n=== Testing different shapes create different cache entries ===")

    # Different shaped arrays should create different cache entries
    a1 = np.array([1, 2, 3, 4], dtype=np.float32)      # Shape: (4,)
    b1 = np.array([5, 6, 7, 8], dtype=np.float32)

    a2 = np.array([1, 2, 3, 4, 5], dtype=np.float32)   # Shape: (5,) - different!
    b2 = np.array([6, 7, 8, 9, 10], dtype=np.float32)

    # First call with shape (4,)
    result1 = cached_add(a1, b1)
    kernel_id_shape4 = get_kernel_id_for_function(cached_add, a1, b1)

    # Second call with shape (5,)
    result2 = cached_add(a2, b2)
    kernel_id_shape5 = get_kernel_id_for_function(cached_add, a2, b2)

    print(f"Kernel ID for shape (4,) arrays: {kernel_id_shape4}")
    print(f"Kernel ID for shape (5,) arrays: {kernel_id_shape5}")

    assert kernel_id_shape4 != kernel_id_shape5, "Different input shapes should create different cache entries"
    print("✅ Different shapes create different cache entries as expected!")


if __name__ == "__main__":
    test_caching_mechanism()
    test_different_shapes_different_cache()