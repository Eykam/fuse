const std = @import("std");
const llvm_c = @import("llvm_c.zig");
const CudaCodegen = @import("cuda_codegen_simple.zig").CudaCodegen;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("CUDA Kernel Code Generator\n", .{});
    std.debug.print("Generating ReLU kernel...\n\n", .{});

    var codegen = try CudaCodegen.init(allocator, "relu_kernel");
    defer codegen.deinit();

    try codegen.generateParallelReLUKernel();

    const ptx = try codegen.emitPTX();
    defer allocator.free(ptx);

    std.debug.print("Generated PTX:\n{s}\n", .{ptx});

    const filename = "relu_kernel.ptx";
    try std.fs.cwd().writeFile(.{ .sub_path = filename, .data = ptx });
    std.debug.print("\nPTX written to {s}\n", .{filename});
}
