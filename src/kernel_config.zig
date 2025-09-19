const std = @import("std");

pub const TensorInfo = struct {
    dtype: []const u8,
    rank: i32,
    shape: []i32,
};

pub const KernelCall = struct {
    id: i32,
    op: []const u8,
    input_ids: []i32,
    output_id: i32,
};

pub const ExternalInput = struct {
    id: i32,
    name: []const u8,
    tensor_info: ?TensorInfo,
    size: usize,  // Size in bytes
};

pub const ExternalOutput = struct {
    id: i32,
    size: usize,  // Size in bytes
};

pub const KernelConfig = struct {
    kernel_calls: []KernelCall,
    external_inputs: []ExternalInput,
    external_outputs: []ExternalOutput,
    return_ids: []i32,
};

pub fn parseConfig(allocator: std.mem.Allocator, json_str: []const u8) !std.json.Parsed(KernelConfig) {
    return try std.json.parseFromSlice(
        KernelConfig,
        allocator,
        json_str,
        .{ .ignore_unknown_fields = true },
    );
}
