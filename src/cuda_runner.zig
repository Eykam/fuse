const std = @import("std");

const cuda = @cImport({
    @cInclude("cuda.h");
});

fn checkCuda(result: cuda.CUresult) !void {
    if (result != cuda.CUDA_SUCCESS) {
        var error_str: [*c]const u8 = undefined;
        _ = cuda.cuGetErrorString(result, &error_str);
        std.debug.print("CUDA error: {s}\n", .{error_str});
        return error.CudaError;
    }
}

fn loadPtxFile(allocator: std.mem.Allocator, filename: []const u8) ![:0]u8 {
    const file = try std.fs.cwd().openFile(filename, .{});
    defer file.close();

    const file_size = try file.getEndPos();
    const ptx = try allocator.allocSentinel(u8, file_size, 0);
    _ = try file.read(ptx);

    return ptx;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len != 2) {
        std.debug.print("Usage: {s} <ptx_file>\n", .{args[0]});
        return error.InvalidArguments;
    }

    // Initialize CUDA
    try checkCuda(cuda.cuInit(0));

    // Get device
    var device: cuda.CUdevice = undefined;
    try checkCuda(cuda.cuDeviceGet(&device, 0));

    // Create context
    var context: cuda.CUcontext = undefined;
    try checkCuda(cuda.cuCtxCreate(&context, 0, device));
    defer _ = cuda.cuCtxDestroy(context);

    // Load PTX
    const ptx = try loadPtxFile(allocator, args[1]);
    defer allocator.free(ptx);

    // Create module from PTX
    var module: cuda.CUmodule = undefined;

    const log_size: usize = 8192;
    const info_log = try allocator.alloc(u8, log_size);
    defer allocator.free(info_log);
    const error_log = try allocator.alloc(u8, log_size);
    defer allocator.free(error_log);

    var options = [_]cuda.CUjit_option{
        cuda.CU_JIT_LOG_VERBOSE,
        cuda.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        cuda.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        cuda.CU_JIT_INFO_LOG_BUFFER,
        cuda.CU_JIT_ERROR_LOG_BUFFER,
    };

    var option_values = [_]?*anyopaque{
        @ptrFromInt(1),
        @ptrFromInt(log_size),
        @ptrFromInt(log_size),
        @ptrCast(info_log.ptr),
        @ptrCast(error_log.ptr),
    };

    const result = cuda.cuModuleLoadDataEx(&module, ptx.ptr, 5, @ptrCast(&options), @ptrCast(&option_values));
    if (result != cuda.CUDA_SUCCESS) {
        std.debug.print("Failed to load PTX module\n", .{});
        std.debug.print("Info log: {s}\n", .{info_log});
        std.debug.print("Error log: {s}\n", .{error_log});
        var error_str: [*c]const u8 = undefined;
        _ = cuda.cuGetErrorString(result, &error_str);
        std.debug.print("CUDA error: {s}\n", .{error_str});
        return error.PtxLoadFailed;
    }
    defer _ = cuda.cuModuleUnload(module);

    std.debug.print("PTX loaded successfully!\n", .{});
    if (std.mem.len(@as([*:0]u8, @ptrCast(info_log.ptr))) > 0) {
        std.debug.print("JIT log:\n{s}\n", .{info_log});
    }

    // Get kernel function
    var kernel: cuda.CUfunction = undefined;
    try checkCuda(cuda.cuModuleGetFunction(&kernel, module, "relu_kernel"));

    // Prepare test data
    const N: i32 = 1024;
    const n_usize = @as(usize, @intCast(N));

    var h_input = try allocator.alloc(f32, n_usize);
    defer allocator.free(h_input);
    const h_output = try allocator.alloc(f32, n_usize);
    defer allocator.free(h_output);
    const h_expected = try allocator.alloc(f32, n_usize);
    defer allocator.free(h_expected);

    // Initialize input with test values (some positive, some negative)
    for (0..n_usize) |i| {
        const i_f32: f32 = @floatFromInt(i);
        const half_n: f32 = @as(f32, @floatFromInt(N)) / 2.0;
        h_input[i] = i_f32 - half_n;  // Range from -512 to 511
        h_expected[i] = @max(0.0, h_input[i]);  // Expected ReLU output
    }

    // Allocate device memory
    var d_input: cuda.CUdeviceptr = undefined;
    var d_output: cuda.CUdeviceptr = undefined;
    try checkCuda(cuda.cuMemAlloc(&d_input, n_usize * @sizeOf(f32)));
    defer _ = cuda.cuMemFree(d_input);
    try checkCuda(cuda.cuMemAlloc(&d_output, n_usize * @sizeOf(f32)));
    defer _ = cuda.cuMemFree(d_output);

    // Copy input to device
    try checkCuda(cuda.cuMemcpyHtoD(d_input, h_input.ptr, n_usize * @sizeOf(f32)));

    // Test multiple configurations
    const configurations = [_]struct { blocks: u32, threads: u32, name: []const u8 }{
        .{ .blocks = 1, .threads = 1, .name = "1 block × 1 thread" },
        .{ .blocks = 1, .threads = 32, .name = "1 block × 32 threads" },
        .{ .blocks = 1, .threads = 64, .name = "1 block × 64 threads" },
        .{ .blocks = 1, .threads = 128, .name = "1 block × 128 threads" },
        .{ .blocks = 1, .threads = 256, .name = "1 block × 256 threads" },
        .{ .blocks = 2, .threads = 256, .name = "2 blocks × 256 threads" },
        .{ .blocks = 4, .threads = 256, .name = "4 blocks × 256 threads" },
        .{ .blocks = 8, .threads = 256, .name = "8 blocks × 256 threads" },
        .{ .blocks = 16, .threads = 64, .name = "16 blocks × 64 threads" },
        .{ .blocks = 32, .threads = 32, .name = "32 blocks × 32 threads" },
    };

    std.debug.print("\n=== Performance Testing ===\n", .{});
    std.debug.print("Testing {} elements with different configurations:\n\n", .{N});

    for (configurations) |config| {
        // Clear output buffer
        try checkCuda(cuda.cuMemsetD32(d_output, 0, n_usize));

        var kernel_params = [_]?*anyopaque{
            @ptrCast(&d_input),
            @ptrCast(&d_output),
            @constCast(@ptrCast(&N)),
        };

        // Warmup run
        try checkCuda(cuda.cuLaunchKernel(
            kernel,
            config.blocks, 1, 1,    // Grid dimensions
            config.threads, 1, 1,   // Block dimensions
            0,                      // Shared memory
            null,                   // Stream
            &kernel_params,
            null,
        ));
        try checkCuda(cuda.cuCtxSynchronize());

        // Timed runs
        const num_runs: u32 = 100;
        const start_time = std.time.nanoTimestamp();

        for (0..num_runs) |_| {
            try checkCuda(cuda.cuLaunchKernel(
                kernel,
                config.blocks, 1, 1,    // Grid dimensions
                config.threads, 1, 1,   // Block dimensions
                0,                      // Shared memory
                null,                   // Stream
                &kernel_params,
                null,
            ));
        }

        try checkCuda(cuda.cuCtxSynchronize());
        const end_time = std.time.nanoTimestamp();

        const total_time_ns = end_time - start_time;
        const avg_time_us = @as(f64, @floatFromInt(total_time_ns)) / @as(f64, @floatFromInt(num_runs)) / 1000.0;
        const total_threads = config.blocks * config.threads;
        const elements_per_second = @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(num_runs)) / (@as(f64, @floatFromInt(total_time_ns)) / 1e9);

        // Copy output back and verify for one run
        try checkCuda(cuda.cuMemcpyDtoH(h_output.ptr, d_output, n_usize * @sizeOf(f32)));

        var errors: u32 = 0;
        for (0..n_usize) |i| {
            if (@abs(h_output[i] - h_expected[i]) > 1e-5) {
                errors += 1;
                if (errors <= 3) {
                    std.debug.print("Error at index {}: expected {d:.2}, got {d:.2}\n", .{ i, h_expected[i], h_output[i] });
                }
            }
        }

        const status = if (errors == 0) "✓" else "✗";
        std.debug.print("{s} {s: <20} | Total threads: {: >4} | Avg time: {d: >8.2}μs | {d: >10.0} elements/sec", .{
            status, config.name, total_threads, avg_time_us, elements_per_second
        });

        if (errors > 0) {
            std.debug.print(" | {} errors", .{errors});
        }
        std.debug.print("\n", .{});
    }

    std.debug.print("\nSample ReLU results:\n", .{});
    const mid: usize = n_usize / 2;
    for ((mid - 5)..(mid + 5)) |i| {
        std.debug.print("  relu({d:7.2}) = {d:7.2}\n", .{ h_input[i], h_output[i] });
    }
}