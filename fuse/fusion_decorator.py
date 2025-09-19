"""
Fusion decorator that extracts kernel info at runtime with actual tensor metadata.
"""

import functools
import inspect
import numpy as np
import ctypes
import os
from typing import Dict, Callable, Optional
from fuse.ast_parser import KernelExtractor, TensorInfo, config_to_json


def extract_tensor_info(arg) -> TensorInfo:
    """Extract tensor metadata from runtime argument"""
    if isinstance(arg, np.ndarray):
        return TensorInfo(
            dtype=str(arg.dtype),  # e.g., "float32", "int64"
            rank=len(arg.shape),  # Number of dimensions
            shape=list(arg.shape),  # Actual shape at runtime
        )
    elif hasattr(arg, "dtype") and hasattr(arg, "shape"):  # PyTorch/TF tensors
        return TensorInfo(
            dtype=str(arg.dtype), rank=len(arg.shape), shape=list(arg.shape)
        )
    else:
        # Scalar or unknown type
        return TensorInfo(dtype=str(type(arg).__name__), rank=0, shape=[])


class FusionResult(ctypes.Structure):
    """FFI result structure from Zig compilation"""

    _fields_ = [
        ("success", ctypes.c_uint8),
        ("kernel_id", ctypes.c_uint64),
        ("error_msg", ctypes.c_char_p),
    ]


class ExecutionResult(ctypes.Structure):
    """FFI result structure from kernel execution"""

    _fields_ = [
        ("success", ctypes.c_uint8),
        ("error_msg", ctypes.c_char_p),
    ]


class KernelFusionFFI:
    """FFI interface to Zig kernel fusion library"""

    def __init__(self):
        lib_path = os.path.join(
            os.path.dirname(__file__), "../zig-out/lib/libkernel_fusion.so"
        )
        if not os.path.exists(lib_path):
            raise RuntimeError(
                f"Kernel fusion library not found at {lib_path}. Run 'zig build ffi' first."
            )

        self.lib = ctypes.CDLL(lib_path)

        # Compilation function
        self.lib.compile_kernel_fusion.argtypes = [ctypes.c_char_p]
        self.lib.compile_kernel_fusion.restype = FusionResult

        # Copy inputs to GPU function
        self.lib.copy_inputs_to_gpu.argtypes = [
            ctypes.c_uint64,                    # kernel_id
            ctypes.POINTER(ctypes.c_char_p),    # input_ptrs
            ctypes.c_uint32,                    # input_count
        ]
        self.lib.copy_inputs_to_gpu.restype = ExecutionResult

        # Execution function
        self.lib.execute_kernel.argtypes = [
            ctypes.c_uint64,                    # kernel_id
            ctypes.c_uint32,                    # element_count
        ]
        self.lib.execute_kernel.restype = ExecutionResult

        # Copy outputs from GPU function
        self.lib.copy_outputs_from_gpu.argtypes = [
            ctypes.c_uint64,                    # kernel_id
            ctypes.POINTER(ctypes.c_char_p),    # output_ptrs
            ctypes.c_uint32,                    # output_count
        ]
        self.lib.copy_outputs_from_gpu.restype = ExecutionResult

        # Cleanup function
        self.lib.free_error_msg.argtypes = [ctypes.c_char_p]

    def compile(self, config_json: str) -> tuple[bool, int, str]:
        """Compile kernel config. Returns (success, kernel_id, error_msg)"""
        config_bytes = config_json.encode("utf-8")
        result = self.lib.compile_kernel_fusion(config_bytes)

        if result.success:
            return True, result.kernel_id, ""
        else:
            error_msg = (
                result.error_msg.decode("utf-8")
                if result.error_msg
                else "Unknown error"
            )
            self.lib.free_error_msg(result.error_msg)
            return False, 0, error_msg

    def execute(self, kernel_id: int, inputs: list, outputs: list) -> tuple[bool, str]:
        """Execute kernel with given inputs and outputs. Returns (success, error_msg)"""
        # Convert numpy arrays to C pointers
        input_ptrs = []
        output_ptrs = []

        # Validate all inputs have same element count
        if not inputs:
            return False, "No input arrays provided"

        element_count = len(inputs[0])
        for i, inp in enumerate(inputs):
            if len(inp) != element_count:
                return False, f"Input {i} size mismatch: expected {element_count}, got {len(inp)}"
            input_ptrs.append(inp.ctypes.data_as(ctypes.c_char_p))

        for i, out in enumerate(outputs):
            if len(out) != element_count:
                return False, f"Output {i} size mismatch: expected {element_count}, got {len(out)}"
            output_ptrs.append(out.ctypes.data_as(ctypes.c_char_p))

        # Convert to C arrays
        input_arr = (ctypes.c_char_p * len(input_ptrs))(*input_ptrs)
        output_arr = (ctypes.c_char_p * len(output_ptrs))(*output_ptrs)

        # Copy inputs to GPU
        result = self.lib.copy_inputs_to_gpu(
            kernel_id,
            input_arr,
            len(input_ptrs)
        )
        if not result.success:
            error_msg = (
                result.error_msg.decode("utf-8")
                if result.error_msg
                else "Unknown error"
            )
            self.lib.free_error_msg(result.error_msg)
            return False, f"Failed to copy inputs to GPU: {error_msg}"

        # Execute kernel on GPU
        result = self.lib.execute_kernel(
            kernel_id,
            element_count
        )
        if not result.success:
            error_msg = (
                result.error_msg.decode("utf-8")
                if result.error_msg
                else "Unknown error"
            )
            self.lib.free_error_msg(result.error_msg)
            return False, f"Failed to execute kernel: {error_msg}"

        # Copy outputs from GPU
        result = self.lib.copy_outputs_from_gpu(
            kernel_id,
            output_arr,
            len(output_ptrs)
        )
        if not result.success:
            error_msg = (
                result.error_msg.decode("utf-8")
                if result.error_msg
                else "Unknown error"
            )
            self.lib.free_error_msg(result.error_msg)
            return False, f"Failed to copy outputs from GPU: {error_msg}"

        return True, ""


_kernel_ffi = None


def get_kernel_ffi():
    """Get or create the global FFI instance"""
    global _kernel_ffi
    if _kernel_ffi is None:
        try:
            _kernel_ffi = KernelFusionFFI()
        except RuntimeError as e:
            print(f"Warning: {e}")
            return None
    return _kernel_ffi


class KernelCacheEntry:
    """Cache entry containing both kernel_id and config"""
    def __init__(self, kernel_id: int, config):
        self.kernel_id = kernel_id
        self.config = config


class FusionCache:
    """Cache for compiled kernels based on tensor metadata"""

    def __init__(self):
        self.cache: Dict[str, KernelCacheEntry] = {}  # cache_key -> KernelCacheEntry

    def get_cache_key(
        self, func_name: str, tensor_infos: Dict[str, TensorInfo]
    ) -> str:
        """Generate cache key from function and tensor metadata"""
        # Create deterministic key from function + arg types/shapes
        key_parts = [func_name]
        for param_name, tensor_info in sorted(tensor_infos.items()):
            key_parts.append(
                f"{param_name}:{tensor_info.dtype}:{tensor_info.rank}:{tensor_info.shape}"
            )
        return "|".join(key_parts)

    def get(self, key: str) -> Optional[KernelCacheEntry]:
        return self.cache.get(key)

    def set(self, key: str, kernel_id: int, config):
        self.cache[key] = KernelCacheEntry(kernel_id, config)


# Global cache instance
_fusion_cache = FusionCache()


def fuse(func: Callable = None, *, strict: bool = False) -> Callable:
    """
    Decorator that enables kernel fusion for the decorated function.
    Extracts kernel operations and tensor metadata at runtime.

    Args:
        strict: If True, raises an exception on compilation/execution errors instead of falling back to Python.
    """
    # Handle both @fuse and @fuse(strict=True) syntax
    if func is None:
        return lambda f: fuse(f, strict=strict)

    # Check if function is defined at module level only
    if "." in func.__qualname__:
        raise RuntimeError(
            f"@fuse decorator on '{func.__qualname__}' is not supported on class methods or nested functions. "
            f"Please use module-level functions only."
        )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get function source and parameter names
        source = inspect.getsource(func)
        sig = inspect.signature(func)
        param_names = list(sig.parameters.keys())

        # Extract tensor info from runtime arguments
        tensor_infos = {}
        for i, (param_name, arg) in enumerate(zip(param_names, args)):
            tensor_infos[param_name] = extract_tensor_info(arg)

        # Generate cache key
        cache_key = _fusion_cache.get_cache_key(func.__name__, tensor_infos)

        ffi = get_kernel_ffi()
        if not ffi:
            if strict:
                raise RuntimeError("FFI not available")
            print("FFI not available, falling back to Python execution")
            return func(*args, **kwargs)

        # Check cache
        cache_entry = _fusion_cache.get(cache_key)

        if cache_entry is None:
            # Cache miss - extract config and compile
            print(f"Cache miss. Extracting kernels...")

            # Extract kernel operations with AST parser
            extractor = KernelExtractor(function_name=func.__name__)
            extractor.runtime_tensor_infos = tensor_infos
            config = extractor.extract_from_source(source)

            # Update external inputs with runtime tensor info and buffer sizes
            for external_input in config.external_inputs:
                var_name = external_input.name.split("::")[-1]
                if var_name in tensor_infos:
                    external_input.tensor_info = tensor_infos[var_name]
                    # Calculate buffer size in bytes
                    if var_name in param_names:
                        arg_index = param_names.index(var_name)
                        if arg_index < len(args):
                            arg = args[arg_index]
                            if hasattr(arg, 'nbytes'):
                                external_input.size = arg.nbytes
                            else:
                                # Default to float32 (4 bytes per element)
                                external_input.size = len(arg) * 4

            # Add external outputs with buffer sizes based on return_ids
            config.external_outputs = []
            for return_id in config.return_ids:
                # For now, assume output has same size as first input
                # TODO: Infer proper size from the operations that produce this return_id
                if args and hasattr(args[0], 'nbytes'):
                    output_size = args[0].nbytes
                else:
                    output_size = len(args[0]) * 4 if args else 0
                config.external_outputs.append({
                    'id': return_id,
                    'size': output_size
                })

            config_json = config_to_json(config)
            print(f"Extracted config with tensor info:")
            print(config_json)

            # Compile kernel
            success, kernel_id, error_msg = ffi.compile(config_json)
            if not success:
                if strict:
                    raise RuntimeError(f"Compilation failed: {error_msg}")
                print(f"Compilation failed: {error_msg}")
                print("Falling back to Python execution")
                return func(*args, **kwargs)

            print(f"Successfully compiled kernel ID: {kernel_id}")
            _fusion_cache.set(cache_key, kernel_id, config)

            # Use the newly created config
            kernel_id = kernel_id
            config = config
        else:
            # Cache hit - use stored kernel_id and config
            print(f"Cache hit! Using kernel ID: {cache_entry.kernel_id}")
            kernel_id = cache_entry.kernel_id
            config = cache_entry.config

        # Prepare inputs based on config.external_inputs order
        inputs = []
        param_names = list(inspect.signature(func).parameters.keys())

        for external_input in config.external_inputs:
            var_name = external_input.name.split("::")[-1]
            if var_name in param_names:
                arg_index = param_names.index(var_name)
                if arg_index < len(args):
                    inputs.append(args[arg_index])
                else:
                    error_msg = f"Error: Missing argument for {var_name}"
                    if strict:
                        raise RuntimeError(error_msg)
                    print(error_msg)
                    print("Falling back to Python execution")
                    return func(*args, **kwargs)
            else:
                error_msg = f"Error: Parameter {var_name} not found in function signature"
                if strict:
                    raise RuntimeError(error_msg)
                print(error_msg)
                print("Falling back to Python execution")
                return func(*args, **kwargs)

        # Prepare outputs based on config.return_ids
        outputs = []
        for return_id in config.return_ids:
            # For now, assume output has same shape/dtype as first input
            # TODO: Infer proper shape from the operations that produce this return_id
            if inputs:
                output = np.empty_like(inputs[0])
                outputs.append(output)
            else:
                error_msg = "Error: No inputs available to infer output shape"
                if strict:
                    raise RuntimeError(error_msg)
                print(error_msg)
                print("Falling back to Python execution")
                return func(*args, **kwargs)

        print(f"Executing kernel with {len(inputs)} inputs and {len(outputs)} outputs")
        success, error_msg = ffi.execute(kernel_id, inputs, outputs)
        if success:
            # Return single output or tuple of outputs
            if len(outputs) == 1:
                return outputs[0]
            else:
                return tuple(outputs)
        else:
            if strict:
                raise RuntimeError(f"Kernel execution failed: {error_msg}")
            print(f"Kernel execution failed: {error_msg}")
            print("Falling back to Python execution")
            return func(*args, **kwargs)

    return wrapper
