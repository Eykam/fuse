const std = @import("std");
const kernel_config = @import("kernel_config.zig");
const KernelCodegen = @import("kernel_codegen.zig").KernelCodegen;

const cuda = @cImport({
    @cInclude("cuda.h");
});

// Global allocator for FFI use
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// Memory manager for persistent GPU buffers
// TODO: Consider using fixed-size arrays instead of dynamic allocation to avoid heap allocations
const MemoryManager = struct {
    input_buffers: []cuda.CUdeviceptr,
    output_buffers: []cuda.CUdeviceptr,
    input_sizes: []usize,
    output_sizes: []usize,
    allocator: std.mem.Allocator,

    const Self = @This();

    fn init(alloc: std.mem.Allocator, input_sizes: []const usize, output_sizes: []const usize) !Self {
        const input_buffers = try alloc.alloc(cuda.CUdeviceptr, input_sizes.len);
        errdefer alloc.free(input_buffers);

        const output_buffers = try alloc.alloc(cuda.CUdeviceptr, output_sizes.len);
        errdefer alloc.free(output_buffers);

        const input_sizes_copy = try alloc.dupe(usize, input_sizes);
        errdefer alloc.free(input_sizes_copy);

        const output_sizes_copy = try alloc.dupe(usize, output_sizes);
        errdefer alloc.free(output_sizes_copy);

        var memory_manager = Self{
            .input_buffers = input_buffers,
            .output_buffers = output_buffers,
            .input_sizes = input_sizes_copy,
            .output_sizes = output_sizes_copy,
            .allocator = alloc,
        };

        // Allocate GPU buffers
        try memory_manager.allocateBuffers();

        return memory_manager;
    }

    fn deinit(self: *Self) void {
        self.freeBuffers();
        self.allocator.free(self.input_buffers);
        self.allocator.free(self.output_buffers);
        self.allocator.free(self.input_sizes);
        self.allocator.free(self.output_sizes);
    }

    fn allocateBuffers(self: *Self) !void {
        // Allocate input buffers with their specific sizes
        for (self.input_buffers, self.input_sizes, 0..) |*buffer, size, i| {
            const result = cuda.cuMemAlloc(buffer, size);
            if (result != cuda.CUDA_SUCCESS) {
                // Cleanup previously allocated input buffers
                for (self.input_buffers[0..i]) |prev_buffer| {
                    _ = cuda.cuMemFree(prev_buffer);
                }
                return error.CudaMemAllocFailed;
            }
        }

        // Allocate output buffers with their specific sizes
        for (self.output_buffers, self.output_sizes, 0..) |*buffer, size, i| {
            const result = cuda.cuMemAlloc(buffer, size);
            if (result != cuda.CUDA_SUCCESS) {
                // Cleanup all input buffers and previously allocated output buffers
                for (self.input_buffers) |input_buffer| {
                    _ = cuda.cuMemFree(input_buffer);
                }
                for (self.output_buffers[0..i]) |prev_buffer| {
                    _ = cuda.cuMemFree(prev_buffer);
                }
                return error.CudaMemAllocFailed;
            }
        }
    }

    fn freeBuffers(self: *Self) void {
        for (self.input_buffers) |buffer| {
            _ = cuda.cuMemFree(buffer);
        }
        for (self.output_buffers) |buffer| {
            _ = cuda.cuMemFree(buffer);
        }
    }

    fn copyInputsToGPU(self: *Self, input_ptrs: [*c][*c]u8) !void {
        for (self.input_buffers, self.input_sizes, 0..) |buffer, size, i| {
            const result = cuda.cuMemcpyHtoD(buffer, input_ptrs[i], size);
            if (result != cuda.CUDA_SUCCESS) {
                return error.CudaMemCopyFailed;
            }
        }
    }

    fn copyOutputsFromGPU(self: *Self, output_ptrs: [*c][*c]u8) !void {
        for (self.output_buffers, self.output_sizes, 0..) |buffer, size, i| {
            const result = cuda.cuMemcpyDtoH(output_ptrs[i], buffer, size);
            if (result != cuda.CUDA_SUCCESS) {
                return error.CudaMemCopyFailed;
            }
        }
    }

    fn getKernelParams(self: *Self, element_count: u32, params: []?*anyopaque) void {
        // Add GPU input buffer pointers
        for (self.input_buffers, 0..) |*buffer, i| {
            params[i] = @ptrCast(@constCast(buffer));
        }

        // Add GPU output buffer pointers
        for (self.output_buffers, 0..) |*buffer, i| {
            params[self.input_buffers.len + i] = @ptrCast(@constCast(buffer));
        }

        // Add element count parameter (number of elements to process)
        params[self.input_buffers.len + self.output_buffers.len] = @ptrCast(@constCast(&element_count));
    }
};

// Launch configuration for CUDA kernels
const LaunchConfig = struct {
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    block_x: u32,
    block_y: u32,
    block_z: u32,
};

// Calculate optimal launch configuration based on kernel operations and tensor shapes
fn calculateLaunchConfig(config: kernel_config.KernelConfig, element_count: u32) LaunchConfig {
    // Analyze kernel operations to determine dimensionality
    var has_2d_ops = false;
    var max_rank: i32 = 0;

    // Check operations for 2D patterns (matmul, conv2d, etc.)
    for (config.kernel_calls) |call| {
        if (std.mem.eql(u8, call.op, "matmul") or
            std.mem.eql(u8, call.op, "conv2d") or
            std.mem.eql(u8, call.op, "bmm")) {
            has_2d_ops = true;
        }
    }

    // Find the maximum tensor rank to determine dimensionality
    for (config.external_inputs) |input| {
        if (input.tensor_info) |tensor_info| {
            max_rank = @max(max_rank, tensor_info.rank);
        }
    }

    // Calculate launch configuration based on operation types and tensor shapes
    if (has_2d_ops and max_rank >= 2) {
        // 2D operations (matrices): Use 2D thread blocks
        return calculate2DLaunchConfig(config, element_count);
    } else {
        // 1D operations (element-wise): Use 1D thread blocks
        return calculate1DLaunchConfig(element_count);
    }
}

// Calculate 1D launch configuration for element-wise operations
fn calculate1DLaunchConfig(element_count: u32) LaunchConfig {
    // Adaptive block size based on problem size
    const block_x: u32 = if (element_count < 256)
        @max(32, element_count)  // Minimum 32 threads, but not more than needed
    else if (element_count < 1024)
        256
    else
        512;  // Larger block for big problems

    const grid_x: u32 = (element_count + block_x - 1) / block_x;

    return LaunchConfig{
        .grid_x = grid_x,
        .grid_y = 1,
        .grid_z = 1,
        .block_x = block_x,
        .block_y = 1,
        .block_z = 1,
    };
}

// Calculate 2D launch configuration for matrix operations
fn calculate2DLaunchConfig(config: kernel_config.KernelConfig, element_count: u32) LaunchConfig {
    // Try to infer matrix dimensions from tensor shapes
    var matrix_height: u32 = 0;
    var matrix_width: u32 = 0;

    // Look for output tensor shape to determine result dimensions
    for (config.external_inputs) |input| {
        if (input.tensor_info) |tensor_info| {
            if (tensor_info.rank >= 2 and tensor_info.shape.len >= 2) {
                // Assume last input is representative of output size
                const h = tensor_info.shape[tensor_info.shape.len - 2];
                const w = tensor_info.shape[tensor_info.shape.len - 1];
                matrix_height = @intCast(h);
                matrix_width = @intCast(w);
            }
        }
    }

    // If we couldn't infer dimensions, fall back to square approximation
    if (matrix_height == 0 or matrix_width == 0) {
        const sqrt_approx = @sqrt(@as(f32, @floatFromInt(element_count)));
        matrix_height = @intFromFloat(@ceil(sqrt_approx));
        matrix_width = matrix_height;
    }

    // Use 32x32 thread blocks for matrix operations (1024 threads total, good occupancy)
    const BLOCK_SIZE: u32 = 32;

    const grid_x: u32 = (matrix_width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const grid_y: u32 = (matrix_height + BLOCK_SIZE - 1) / BLOCK_SIZE;

    return LaunchConfig{
        .grid_x = grid_x,
        .grid_y = grid_y,
        .grid_z = 1,
        .block_x = BLOCK_SIZE,
        .block_y = BLOCK_SIZE,
        .block_z = 1,
    };
}

// Kernel cache: hash -> compiled CUDA module + function + memory manager
const KernelEntry = struct {
    module: cuda.CUmodule,
    function: cuda.CUfunction,
    ptx_code: []u8,
    memory_manager: MemoryManager,
    config: kernel_config.KernelConfig,  // Store config for launch calculations
};

var kernel_cache: std.AutoHashMap(u64, KernelEntry) = undefined;
var cache_initialized = false;
var cuda_context: cuda.CUcontext = undefined;
var cuda_initialized = false;

// Result structure to return to Python (must be extern for FFI)
pub const FusionResult = extern struct {
    success: u8,  // Use u8 instead of bool for FFI compatibility
    kernel_id: u64,
    error_msg: [*c]const u8,
};

// Execution result structure
pub const ExecutionResult = extern struct {
    success: u8,  // Use u8 instead of bool for FFI compatibility
    error_msg: [*c]const u8,
};

// Helper functions
fn checkCuda(result: cuda.CUresult) !void {
    if (result != cuda.CUDA_SUCCESS) {
        var error_str: [*c]const u8 = undefined;
        _ = cuda.cuGetErrorString(result, &error_str);
        std.debug.print("CUDA error: {s}\n", .{error_str});
        return error.CudaError;
    }
}

fn initCuda() !void {
    if (cuda_initialized) return;

    std.debug.print("DEBUG: Initializing CUDA...\n", .{});
    try checkCuda(cuda.cuInit(0));
    std.debug.print("DEBUG: CUDA initialized\n", .{});

    var device: cuda.CUdevice = undefined;
    std.debug.print("DEBUG: Getting CUDA device...\n", .{});
    try checkCuda(cuda.cuDeviceGet(&device, 0));
    std.debug.print("DEBUG: Got CUDA device: {}\n", .{device});

    std.debug.print("DEBUG: Creating CUDA context...\n", .{});
    try checkCuda(cuda.cuCtxCreate(&cuda_context, 0, device));
    std.debug.print("DEBUG: CUDA context created\n", .{});
    cuda_initialized = true;
}

fn initCache() !void {
    if (cache_initialized) return;

    kernel_cache = std.AutoHashMap(u64, KernelEntry).init(allocator);
    cache_initialized = true;
}

fn hashConfig(config_str: []const u8) u64 {
    return std.hash_map.hashString(config_str);
}

// FFI export: Compile kernel config into PTX
export fn compile_kernel_fusion(json_config: [*c]const u8) FusionResult {
    // Initialize CUDA and cache if needed
    initCuda() catch |err| {
        const error_msg = std.fmt.allocPrintZ(allocator, "Failed to initialize CUDA: {}", .{err}) catch "CUDA init error";
        return FusionResult{
            .success = 0,
            .kernel_id = 0,
            .error_msg = error_msg.ptr,
        };
    };

    initCache() catch |err| {
        const error_msg = std.fmt.allocPrintZ(allocator, "Failed to initialize cache: {}", .{err}) catch "Cache init error";
        return FusionResult{
            .success = 0,
            .kernel_id = 0,
            .error_msg = error_msg.ptr,
        };
    };

    // Convert C string to Zig slice
    const config_str = std.mem.sliceTo(json_config, 0);

    // Generate hash for this config
    const kernel_id = hashConfig(config_str);

    // Check if already cached
    if (kernel_cache.get(kernel_id)) |cached| {
        _ = cached;
        std.debug.print("Cache hit for kernel ID: {}\n", .{kernel_id});
        return FusionResult{
            .success = 1,
            .kernel_id = kernel_id,
            .error_msg = null,
        };
    }

    std.debug.print("Cache miss for kernel ID: {}\n", .{kernel_id});

    // Parse JSON config
    const parsed = kernel_config.parseConfig(allocator, config_str) catch |err| {
        const error_msg = std.fmt.allocPrintZ(allocator, "Failed to parse JSON config: {}", .{err}) catch "Parse error";
        return FusionResult{
            .success = 0,
            .kernel_id = 0,
            .error_msg = error_msg.ptr,
        };
    };
    defer parsed.deinit();

    const config = parsed.value;

    // Debug: Print parsed config
    std.debug.print("{s}\n", .{"=" ** 80});
    std.debug.print("From Zig...\n", .{});
    std.debug.print("Parsed kernel config:\n", .{});
    std.debug.print("  Kernel calls: {d}\n", .{config.kernel_calls.len});
    for (config.kernel_calls) |call| {
        std.debug.print("    - {s} (id={d})\n", .{ call.op, call.id });
    }
    std.debug.print("  External inputs: {d}\n", .{config.external_inputs.len});
    for (config.external_inputs) |input| {
        std.debug.print("    - id={d}, size={d} bytes\n", .{ input.id, input.size });
    }
    std.debug.print("  External outputs: {d}\n", .{config.external_outputs.len});
    for (config.external_outputs) |output| {
        std.debug.print("    - id={d}, size={d} bytes\n", .{ output.id, output.size });
    }
    std.debug.print("  Return IDs: {any}\n", .{config.return_ids});
    std.debug.print("{s}\n\n", .{"=" ** 80});

    // Generate fused kernel based on config
    var codegen = KernelCodegen.init(allocator, "fused_kernel") catch |err| {
        const error_msg = std.fmt.allocPrintZ(allocator, "Failed to initialize kernel codegen: {}", .{err}) catch "Codegen init error";
        return FusionResult{
            .success = 0,
            .kernel_id = 0,
            .error_msg = error_msg.ptr,
        };
    };
    defer codegen.deinit();

    // Create the fused kernel
    _ = codegen.createFusedKernel(&config) catch |err| {
        const error_msg = std.fmt.allocPrintZ(allocator, "Failed to create fused kernel: {}", .{err}) catch "Kernel creation error";
        return FusionResult{
            .success = 0,
            .kernel_id = 0,
            .error_msg = error_msg.ptr,
        };
    };

    // Emit PTX code
    const ptx_code = codegen.emitPTX() catch |err| {
        const error_msg = std.fmt.allocPrintZ(allocator, "Failed to emit PTX: {}", .{err}) catch "PTX emission error";
        return FusionResult{
            .success = 0,
            .kernel_id = 0,
            .error_msg = error_msg.ptr,
        };
    };

    // Write PTX to file for debugging
    const ptx_filename = std.fmt.allocPrintZ(allocator, "fused_kernel_{x}.ptx", .{kernel_id}) catch {
        allocator.free(ptx_code);
        const error_msg = "Failed to create PTX filename";
        return FusionResult{
            .success = 0,
            .kernel_id = 0,
            .error_msg = error_msg.ptr,
        };
    };
    defer allocator.free(ptx_filename);

    std.fs.cwd().writeFile(.{ .sub_path = ptx_filename, .data = ptx_code }) catch |err| {
        std.debug.print("Warning: Failed to write PTX file {s}: {}\n", .{ ptx_filename, err });
    };
    std.debug.print("PTX written to: {s}\n", .{ptx_filename});

    // Compile PTX to CUDA module
    std.debug.print("DEBUG: About to load PTX module...\n", .{});
    var module: cuda.CUmodule = undefined;
    var function: cuda.CUfunction = undefined;

    std.debug.print("DEBUG: Calling cuModuleLoadData...\n", .{});
    const load_result = cuda.cuModuleLoadData(&module, ptx_code.ptr);
    std.debug.print("DEBUG: cuModuleLoadData returned: {}\n", .{load_result});
    if (load_result != cuda.CUDA_SUCCESS) {
        allocator.free(ptx_code);
        const error_msg = std.fmt.allocPrintZ(allocator, "Failed to load PTX module: CUDA error {}", .{load_result}) catch "PTX load error";
        return FusionResult{
            .success = 0,
            .kernel_id = 0,
            .error_msg = error_msg.ptr,
        };
    }

    std.debug.print("DEBUG: Calling cuModuleGetFunction...\n", .{});
    const func_result = cuda.cuModuleGetFunction(&function, module, "fused_kernel");
    std.debug.print("DEBUG: cuModuleGetFunction returned: {}\n", .{func_result});
    if (func_result != cuda.CUDA_SUCCESS) {
        _ = cuda.cuModuleUnload(module);
        allocator.free(ptx_code);
        const error_msg = std.fmt.allocPrintZ(allocator, "Failed to get kernel function: CUDA error {}", .{func_result}) catch "Function get error";
        return FusionResult{
            .success = 0,
            .kernel_id = 0,
            .error_msg = error_msg.ptr,
        };
    }
    std.debug.print("DEBUG: Successfully loaded kernel function!\n", .{});

    // Extract buffer sizes from config
    const input_sizes = allocator.alloc(usize, config.external_inputs.len) catch |err| {
        _ = cuda.cuModuleUnload(module);
        allocator.free(ptx_code);
        const error_msg = std.fmt.allocPrintZ(allocator, "Failed to allocate input sizes: {}", .{err}) catch "Allocation error";
        return FusionResult{
            .success = 0,
            .kernel_id = 0,
            .error_msg = error_msg.ptr,
        };
    };
    defer allocator.free(input_sizes);

    for (config.external_inputs, 0..) |input, i| {
        input_sizes[i] = input.size;
    }

    const output_sizes = allocator.alloc(usize, config.external_outputs.len) catch |err| {
        _ = cuda.cuModuleUnload(module);
        allocator.free(ptx_code);
        const error_msg = std.fmt.allocPrintZ(allocator, "Failed to allocate output sizes: {}", .{err}) catch "Allocation error";
        return FusionResult{
            .success = 0,
            .kernel_id = 0,
            .error_msg = error_msg.ptr,
        };
    };
    defer allocator.free(output_sizes);

    for (config.external_outputs, 0..) |output, i| {
        output_sizes[i] = output.size;
    }

    // Create memory manager with buffer sizes from config
    const memory_manager = MemoryManager.init(allocator, input_sizes, output_sizes) catch |err| {
        _ = cuda.cuModuleUnload(module);
        allocator.free(ptx_code);
        const error_msg = std.fmt.allocPrintZ(allocator, "Failed to create memory manager: {}", .{err}) catch "Memory manager error";
        return FusionResult{
            .success = 0,
            .kernel_id = 0,
            .error_msg = error_msg.ptr,
        };
    };

    // Store in cache with memory manager and config
    const kernel_entry = KernelEntry{
        .module = module,
        .function = function,
        .ptx_code = ptx_code,
        .memory_manager = memory_manager,
        .config = config,
    };

    kernel_cache.put(kernel_id, kernel_entry) catch |err| {
        var mm = memory_manager;
        mm.deinit();
        _ = cuda.cuModuleUnload(module);
        allocator.free(ptx_code);
        const error_msg = std.fmt.allocPrintZ(allocator, "Failed to cache kernel: {}", .{err}) catch "Cache error";
        return FusionResult{
            .success = 0,
            .kernel_id = 0,
            .error_msg = error_msg.ptr,
        };
    };

    std.debug.print("Successfully compiled and cached kernel ID: {}\n", .{kernel_id});

    return FusionResult{
        .success = 1,
        .kernel_id = kernel_id,
        .error_msg = null,
    };
}

// FFI export: Copy inputs to GPU for a kernel
export fn copy_inputs_to_gpu(
    kernel_id: u64,
    input_ptrs: [*c][*c]u8,
    input_count: u32,
) ExecutionResult {
    // Get cached kernel
    var kernel_entry = kernel_cache.getPtr(kernel_id) orelse {
        const error_msg = std.fmt.allocPrintZ(allocator, "Kernel ID {} not found in cache", .{kernel_id}) catch "Kernel not found";
        return ExecutionResult{
            .success = 0,
            .error_msg = error_msg.ptr,
        };
    };

    // Validate input count
    if (input_count != kernel_entry.memory_manager.input_buffers.len) {
        const error_msg = std.fmt.allocPrintZ(allocator, "Input count mismatch: expected {}, got {}", .{ kernel_entry.memory_manager.input_buffers.len, input_count }) catch "Input count error";
        return ExecutionResult{
            .success = 0,
            .error_msg = error_msg.ptr,
        };
    }

    // Copy inputs to GPU
    kernel_entry.memory_manager.copyInputsToGPU(input_ptrs) catch |err| {
        const error_msg = std.fmt.allocPrintZ(allocator, "Failed to copy inputs to GPU: {}", .{err}) catch "Copy error";
        return ExecutionResult{
            .success = 0,
            .error_msg = error_msg.ptr,
        };
    };

    return ExecutionResult{
        .success = 1,
        .error_msg = null,
    };
}

// FFI export: Execute kernel with given ID
export fn execute_kernel(
    kernel_id: u64,
    element_count: u32,
) ExecutionResult {
    // Get cached kernel
    var kernel_entry = kernel_cache.getPtr(kernel_id) orelse {
        const error_msg = std.fmt.allocPrintZ(allocator, "Kernel ID {} not found in cache", .{kernel_id}) catch "Kernel not found";
        return ExecutionResult{
            .success = 0,
            .error_msg = error_msg.ptr,
        };
    };

    // Prepare kernel parameters using MemoryManager
    const total_params = kernel_entry.memory_manager.input_buffers.len +
                         kernel_entry.memory_manager.output_buffers.len + 1;
    const params = allocator.alloc(?*anyopaque, total_params) catch {
        const error_msg = "Failed to allocate parameter array";
        return ExecutionResult{
            .success = 0,
            .error_msg = error_msg.ptr,
        };
    };
    defer allocator.free(params);

    // Let MemoryManager set up the kernel parameters
    kernel_entry.memory_manager.getKernelParams(element_count, params);

    // Calculate optimal launch configuration based on kernel operations and tensor shapes
    const launch_config = calculateLaunchConfig(kernel_entry.config, element_count);

    std.debug.print("DEBUG: Using launch config - Grid: ({d},{d},{d}), Block: ({d},{d},{d})\n", .{
        launch_config.grid_x, launch_config.grid_y, launch_config.grid_z,
        launch_config.block_x, launch_config.block_y, launch_config.block_z,
    });

    const launch_result = cuda.cuLaunchKernel(
        kernel_entry.function,
        launch_config.grid_x, launch_config.grid_y, launch_config.grid_z,  // Grid dimensions
        launch_config.block_x, launch_config.block_y, launch_config.block_z, // Block dimensions
        0,                   // Shared memory size
        null,                // Stream
        @ptrCast(params.ptr), // Parameters
        null,                // Extra parameters
    );

    if (launch_result != cuda.CUDA_SUCCESS) {
        const error_msg = std.fmt.allocPrintZ(allocator, "Kernel launch failed: CUDA error {}", .{launch_result}) catch "Launch error";
        return ExecutionResult{
            .success = 0,
            .error_msg = error_msg.ptr,
        };
    }

    // Synchronize to wait for kernel completion
    const sync_result = cuda.cuCtxSynchronize();
    if (sync_result != cuda.CUDA_SUCCESS) {
        const error_msg = std.fmt.allocPrintZ(allocator, "Kernel synchronization failed: CUDA error {}", .{sync_result}) catch "Sync error";
        return ExecutionResult{
            .success = 0,
            .error_msg = error_msg.ptr,
        };
    }

    return ExecutionResult{
        .success = 1,
        .error_msg = null,
    };
}

// FFI export: Copy outputs from GPU for a kernel
export fn copy_outputs_from_gpu(
    kernel_id: u64,
    output_ptrs: [*c][*c]u8,
    output_count: u32,
) ExecutionResult {
    // Get cached kernel
    var kernel_entry = kernel_cache.getPtr(kernel_id) orelse {
        const error_msg = std.fmt.allocPrintZ(allocator, "Kernel ID {} not found in cache", .{kernel_id}) catch "Kernel not found";
        return ExecutionResult{
            .success = 0,
            .error_msg = error_msg.ptr,
        };
    };

    // Validate output count
    if (output_count != kernel_entry.memory_manager.output_buffers.len) {
        const error_msg = std.fmt.allocPrintZ(allocator, "Output count mismatch: expected {}, got {}", .{ kernel_entry.memory_manager.output_buffers.len, output_count }) catch "Output count error";
        return ExecutionResult{
            .success = 0,
            .error_msg = error_msg.ptr,
        };
    }

    // Copy outputs from GPU
    kernel_entry.memory_manager.copyOutputsFromGPU(output_ptrs) catch |err| {
        const error_msg = std.fmt.allocPrintZ(allocator, "Failed to copy outputs from GPU: {}", .{err}) catch "Copy error";
        return ExecutionResult{
            .success = 0,
            .error_msg = error_msg.ptr,
        };
    };

    return ExecutionResult{
        .success = 1,
        .error_msg = null,
    };
}

// FFI export: Free error message
export fn free_error_msg(error_msg: [*c]u8) void {
    // Note: Some error messages are string literals, others are allocated
    // For safety, we'll avoid freeing since the GPA will clean up at shutdown
    // TODO: Better approach would be to track which messages are allocated
    _ = error_msg;
}
