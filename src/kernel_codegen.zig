const std = @import("std");
const llvm = @import("llvm_c.zig");

// Target configuration constants
const NVPTX_TARGET_TRIPLE = "nvptx64-nvidia-cuda";
const NVPTX_DATA_LAYOUT = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64";
// TODO: Don't hardcode compute capability - make it configurable
const NVPTX_CPU = "sm_75"; // Targeting compute capability 7.5
const NVPTX_FEATURES = ""; // Empty string means use default features for the target

pub const DType = enum {
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,

    pub fn fromString(str: []const u8) !DType {
        if (std.mem.eql(u8, str, "float32")) return .float32;
        if (std.mem.eql(u8, str, "float64")) return .float64;
        if (std.mem.eql(u8, str, "int8")) return .int8;
        if (std.mem.eql(u8, str, "int16")) return .int16;
        if (std.mem.eql(u8, str, "int32")) return .int32;
        if (std.mem.eql(u8, str, "int64")) return .int64;
        return error.UnsupportedDtype;
    }

    pub fn toLLVMType(self: DType, context: *llvm.Context) *llvm.Type {
        return switch (self) {
            .float32 => llvm.Type.floatType(context),
            .float64 => llvm.Type.doubleType(context),
            .int8 => llvm.Type.int8Type(context),
            .int16 => llvm.Type.int16Type(context),
            .int32 => llvm.Type.int32Type(context),
            .int64 => llvm.Type.int64Type(context),
        };
    }

    pub fn isFloatingPoint(self: DType) bool {
        return switch (self) {
            .float32, .float64 => true,
            else => false,
        };
    }

    pub fn isInteger(self: DType) bool {
        return !self.isFloatingPoint();
    }
};

pub const Operation = enum {
    add,
    subtract,
    multiply,
    divide,
    relu,

    pub fn fromString(str: []const u8) !Operation {
        if (std.mem.eql(u8, str, "add")) return .add;
        if (std.mem.eql(u8, str, "subtract")) return .subtract;
        if (std.mem.eql(u8, str, "multiply")) return .multiply;
        if (std.mem.eql(u8, str, "divide")) return .divide;
        if (std.mem.eql(u8, str, "relu")) return .relu;
        return error.UnknownOperation;
    }

    pub fn execute(self: Operation, codegen: *KernelCodegen, inputs: []*llvm.Value) !*llvm.Value {
        return switch (self) {
            .add => blk: {
                if (inputs.len != 2) return error.InvalidInputCount;
                break :blk codegen.generateAddOp(inputs[0], inputs[1]);
            },
            .subtract => blk: {
                if (inputs.len != 2) return error.InvalidInputCount;
                break :blk codegen.generateSubtractOp(inputs[0], inputs[1]);
            },
            .multiply => blk: {
                if (inputs.len != 2) return error.InvalidInputCount;
                break :blk codegen.generateMultiplyOp(inputs[0], inputs[1]);
            },
            .divide => blk: {
                if (inputs.len != 2) return error.InvalidInputCount;
                break :blk codegen.generateDivideOp(inputs[0], inputs[1]);
            },
            .relu => blk: {
                if (inputs.len != 1) return error.InvalidInputCount;
                break :blk codegen.generateReLUOp(inputs[0]);
            },
        };
    }
};

pub const KernelCodegen = struct {
    allocator: std.mem.Allocator,
    context: *llvm.Context,
    module: *llvm.Module,
    builder: *llvm.Builder,
    kernel_name: []const u8,

    pub fn init(allocator: std.mem.Allocator, kernel_name: []const u8) !KernelCodegen {
        llvm.initializeNVPTXTarget();

        const context = llvm.Context.create();
        const module_name = try std.fmt.allocPrintZ(allocator, "{s}_module", .{kernel_name});
        defer allocator.free(module_name);

        const module = llvm.Module.create(module_name, context);

        module.setDataLayout(NVPTX_DATA_LAYOUT);
        module.setTargetTriple(NVPTX_TARGET_TRIPLE);

        const builder = llvm.Builder.create(context);

        return KernelCodegen{
            .allocator = allocator,
            .context = context,
            .module = module,
            .builder = builder,
            .kernel_name = kernel_name,
        };
    }

    pub fn deinit(self: *KernelCodegen) void {
        self.builder.destroy();
        self.module.destroy();
        self.context.destroy();
    }

    pub fn generateAddOp(self: *KernelCodegen, a_val: *llvm.Value, b_val: *llvm.Value) *llvm.Value {
        return self.builder.buildFAdd(a_val, b_val, "add.result");
    }

    pub fn generateMultiplyOp(self: *KernelCodegen, a_val: *llvm.Value, b_val: *llvm.Value) *llvm.Value {
        return self.builder.buildFMul(a_val, b_val, "mul.result");
    }

    pub fn generateReLUOp(self: *KernelCodegen, input_val: *llvm.Value) *llvm.Value {
        const float_type = llvm.Type.floatType(self.context);
        const zero = llvm.Value.constFloat(float_type, 0.0);
        const cmp = self.builder.buildFCmpOGT(input_val, zero, "cmp");
        return self.builder.buildSelect(cmp, input_val, zero, "relu.result");
    }

    pub fn generateSubtractOp(self: *KernelCodegen, a_val: *llvm.Value, b_val: *llvm.Value) *llvm.Value {
        return self.builder.buildFSub(a_val, b_val, "sub.result");
    }

    pub fn generateDivideOp(self: *KernelCodegen, a_val: *llvm.Value, b_val: *llvm.Value) *llvm.Value {
        return self.builder.buildFDiv(a_val, b_val, "div.result");
    }

    fn getThreadIdX(self: *KernelCodegen) struct { func: *llvm.Value, type: *llvm.Type } {
        const int32_type = llvm.Type.int32Type(self.context);
        const func_type = llvm.Type.functionType(int32_type, &[_]*llvm.Type{}, false);
        const func = llvm.Function.addFunction(self.module, "llvm.nvvm.read.ptx.sreg.tid.x", func_type);
        return .{ .func = func.asValue(), .type = func_type };
    }

    fn getBlockDimX(self: *KernelCodegen) struct { func: *llvm.Value, type: *llvm.Type } {
        const int32_type = llvm.Type.int32Type(self.context);
        const func_type = llvm.Type.functionType(int32_type, &[_]*llvm.Type{}, false);
        const func = llvm.Function.addFunction(self.module, "llvm.nvvm.read.ptx.sreg.ntid.x", func_type);
        return .{ .func = func.asValue(), .type = func_type };
    }

    fn getBlockIdxX(self: *KernelCodegen) struct { func: *llvm.Value, type: *llvm.Type } {
        const int32_type = llvm.Type.int32Type(self.context);
        const func_type = llvm.Type.functionType(int32_type, &[_]*llvm.Type{}, false);
        const func = llvm.Function.addFunction(self.module, "llvm.nvvm.read.ptx.sreg.ctaid.x", func_type);
        return .{ .func = func.asValue(), .type = func_type };
    }

    fn getIntAddFunc(self: *KernelCodegen) *llvm.Value {
        const int32_type = llvm.Type.int32Type(self.context);
        const func_type = llvm.Type.functionType(int32_type, &[_]*llvm.Type{ int32_type, int32_type }, false);
        const func = llvm.Function.addFunction(self.module, "__add_i32", func_type);

        const entry = func.appendBasicBlock("entry");
        const saved_builder = self.builder;
        const temp_builder = llvm.Builder.create(self.context);
        defer temp_builder.destroy();

        temp_builder.setInsertPoint(entry);
        const a = func.asValue().getParam(0);
        const b = func.asValue().getParam(1);
        const result = @as(*llvm.Value, @ptrCast(llvm.c.LLVMBuildAdd(@ptrCast(temp_builder), @ptrCast(a), @ptrCast(b), "sum")));
        _ = temp_builder.buildRet(result);

        self.builder = saved_builder;
        return func.asValue();
    }

    fn getIntMulFunc(self: *KernelCodegen) *llvm.Value {
        const int32_type = llvm.Type.int32Type(self.context);
        const func_type = llvm.Type.functionType(int32_type, &[_]*llvm.Type{ int32_type, int32_type }, false);
        const func = llvm.Function.addFunction(self.module, "__mul_i32", func_type);

        const entry = func.appendBasicBlock("entry");
        const saved_builder = self.builder;
        const temp_builder = llvm.Builder.create(self.context);
        defer temp_builder.destroy();

        temp_builder.setInsertPoint(entry);
        const a = func.asValue().getParam(0);
        const b = func.asValue().getParam(1);
        const result = @as(*llvm.Value, @ptrCast(llvm.c.LLVMBuildMul(@ptrCast(temp_builder), @ptrCast(a), @ptrCast(b), "product")));
        _ = temp_builder.buildRet(result);

        self.builder = saved_builder;
        return func.asValue();
    }

    fn getIntCmpFunc(self: *KernelCodegen) *llvm.Value {
        const int32_type = llvm.Type.int32Type(self.context);
        const func_type = llvm.Type.functionType(int32_type, &[_]*llvm.Type{ int32_type, int32_type }, false);
        const func = llvm.Function.addFunction(self.module, "__cmp_lt_i32", func_type);

        const entry = func.appendBasicBlock("entry");
        const saved_builder = self.builder;
        const temp_builder = llvm.Builder.create(self.context);
        defer temp_builder.destroy();

        temp_builder.setInsertPoint(entry);
        const a = func.asValue().getParam(0);
        const b = func.asValue().getParam(1);
        const cmp = @as(*llvm.Value, @ptrCast(llvm.c.LLVMBuildICmp(@ptrCast(temp_builder), llvm.c.LLVMIntSLT, @ptrCast(a), @ptrCast(b), "cmp")));
        const result = @as(*llvm.Value, @ptrCast(llvm.c.LLVMBuildZExt(@ptrCast(temp_builder), @ptrCast(cmp), @ptrCast(int32_type), "cmp.i32")));
        _ = temp_builder.buildRet(result);

        self.builder = saved_builder;
        return func.asValue();
    }

    fn getCondBrFunc(self: *KernelCodegen) *llvm.Value {
        const void_type = llvm.Type.voidType(self.context);
        const int32_type = llvm.Type.int32Type(self.context);
        const block_ptr_type = llvm.Type.pointerType(llvm.Type.int32Type(self.context), 0);
        const func_type = llvm.Type.functionType(void_type, &[_]*llvm.Type{ int32_type, block_ptr_type, block_ptr_type }, false);
        const func = llvm.Function.addFunction(self.module, "__cond_br", func_type);

        const entry = func.appendBasicBlock("entry");
        const saved_builder = self.builder;
        const temp_builder = llvm.Builder.create(self.context);
        defer temp_builder.destroy();

        temp_builder.setInsertPoint(entry);
        const cond = func.asValue().getParam(0);
        const true_block = func.asValue().getParam(1);
        const false_block = func.asValue().getParam(2);

        const cond_bool = @as(*llvm.Value, @ptrCast(llvm.c.LLVMBuildICmp(@ptrCast(temp_builder), llvm.c.LLVMIntNE, @ptrCast(cond), @ptrCast(llvm.Value.constInt(int32_type, 0, false)), "cond.bool")));

        _ = @as(*llvm.Value, @ptrCast(llvm.c.LLVMBuildCondBr(@ptrCast(temp_builder), @ptrCast(cond_bool), @ptrCast(true_block), @ptrCast(false_block))));

        self.builder = saved_builder;
        return func.asValue();
    }

    fn getBrFunc(self: *KernelCodegen) *llvm.Value {
        const void_type = llvm.Type.voidType(self.context);
        const block_ptr_type = llvm.Type.pointerType(llvm.Type.int32Type(self.context), 0);
        const func_type = llvm.Type.functionType(void_type, &[_]*llvm.Type{block_ptr_type}, false);
        const func = llvm.Function.addFunction(self.module, "__br", func_type);

        const entry = func.appendBasicBlock("entry");
        const saved_builder = self.builder;
        const temp_builder = llvm.Builder.create(self.context);
        defer temp_builder.destroy();

        temp_builder.setInsertPoint(entry);
        const block = func.asValue().getParam(0);

        _ = @as(*llvm.Value, @ptrCast(llvm.c.LLVMBuildBr(@ptrCast(temp_builder), @ptrCast(block))));

        self.builder = saved_builder;
        return func.asValue();
    }

    pub fn createFusedKernel(self: *KernelCodegen, config: *const @import("kernel_config.zig").KernelConfig) !*llvm.Function {
        const void_type = llvm.Type.voidType(self.context);
        const int32_type = llvm.Type.int32Type(self.context);

        var param_types = std.ArrayList(*llvm.Type).init(self.allocator);
        defer param_types.deinit();

        // Input arrays
        for (config.external_inputs) |input| {
            const dtype = try DType.fromString(input.tensor_info.?.dtype);
            const base_type = dtype.toLLVMType(self.context);
            const ptr_type = llvm.Type.pointerType(base_type, 1);
            try param_types.append(ptr_type);
        }

        // TODO: Handle external inputs with different shapes - may need per-array size params
        // For now, assuming all inputs have same shape (validated in Python)

        // Output buffers - one for each return_id
        // return_ids already contains only the values actually returned from the function
        for (config.return_ids) |_| {
            // TODO: Determine output dtype from the operation that produces this result
            // For now, assume same dtype as first input
            const output_dtype = try DType.fromString(config.external_inputs[0].tensor_info.?.dtype);
            const output_base_type = output_dtype.toLLVMType(self.context);
            const output_ptr_type = llvm.Type.pointerType(output_base_type, 1);
            try param_types.append(output_ptr_type);
        }

        // Single size parameter (assuming all arrays are same size for element-wise ops)
        try param_types.append(int32_type);

        const func_type = llvm.Type.functionType(void_type, param_types.items, false);
        const kernel_name_z = try std.fmt.allocPrintZ(self.allocator, "{s}", .{self.kernel_name});
        defer self.allocator.free(kernel_name_z);
        const kernel_func = llvm.Function.addFunction(self.module, kernel_name_z, func_type);

        kernel_func.setLinkage(llvm.c.LLVMExternalLinkage);
        kernel_func.setCallingConv(llvm.c.LLVMPTXKernelCallConv);

        const entry_block = kernel_func.appendBasicBlock("entry");
        const compute_block = kernel_func.appendBasicBlock("compute");
        const exit_block = kernel_func.appendBasicBlock("exit");

        self.builder.setInsertPoint(entry_block);

        // Calculate global thread index
        const tid_x_info = self.getThreadIdX();
        const tid_x = self.builder.buildCall(tid_x_info.type, tid_x_info.func, &[_]*llvm.Value{}, "tid.x");

        const bdim_x_info = self.getBlockDimX();
        const bdim_x = self.builder.buildCall(bdim_x_info.type, bdim_x_info.func, &[_]*llvm.Value{}, "bdim.x");

        const bid_x_info = self.getBlockIdxX();
        const bid_x = self.builder.buildCall(bid_x_info.type, bid_x_info.func, &[_]*llvm.Value{}, "bid.x");

        const bid_mul_bdim = self.builder.buildMul(bid_x, bdim_x, "bid.mul.bdim");
        const global_idx = self.builder.buildAdd(bid_mul_bdim, tid_x, "global.idx");

        // Bounds check - Python validates all arrays have same size
        const size_param = kernel_func.asValue().getParam(@intCast(param_types.items.len - 1));
        const in_bounds = self.builder.buildICmpULT(global_idx, size_param, "in.bounds");

        _ = self.builder.buildCondBr(in_bounds, compute_block, exit_block);

        self.builder.setInsertPoint(exit_block);
        _ = self.builder.buildRetVoid();

        self.builder.setInsertPoint(compute_block);

        // Load values from input arrays at global_idx
        var input_values = std.ArrayList(*llvm.Value).init(self.allocator);
        defer input_values.deinit();

        for (config.external_inputs, 0..) |input, i| {
            const input_ptr = kernel_func.asValue().getParam(@intCast(i));
            const dtype = try DType.fromString(input.tensor_info.?.dtype);
            const element_type = dtype.toLLVMType(self.context);
            const element_ptr = self.builder.buildGEP(element_type, input_ptr, &[_]*llvm.Value{global_idx}, "input.ptr");
            const loaded_value = self.builder.buildLoad(element_type, element_ptr, "input.val");
            try input_values.append(loaded_value);
        }

        // Execute operations based on config.kernel_calls
        // Map from value ID to LLVM value (external inputs have negative IDs, kernel outputs positive)
        var value_map = std.AutoHashMap(i32, *llvm.Value).init(self.allocator);
        defer value_map.deinit();

        // Store external input values in the map with their negative IDs
        for (config.external_inputs, 0..) |input, i| {
            try value_map.put(input.id, input_values.items[i]);
        }

        // Execute kernel operations in order
        for (config.kernel_calls) |kernel_call| {
            // Get input values for this operation
            var op_inputs = std.ArrayList(*llvm.Value).init(self.allocator);
            defer op_inputs.deinit();

            for (kernel_call.input_ids) |input_id| {
                const input_value = value_map.get(input_id) orelse {
                    std.debug.print("Error: Value with ID {} not found\n", .{input_id});
                    return error.ValueNotFound;
                };
                try op_inputs.append(input_value);
            }

            // Execute the operation
            const operation = try Operation.fromString(kernel_call.op);
            const result_value = try operation.execute(self, op_inputs.items);

            // Store result in value map
            try value_map.put(kernel_call.output_id, result_value);
        }

        // Store results to output arrays at global_idx
        const num_inputs = config.external_inputs.len;
        for (config.return_ids, 0..) |return_id, i| {
            const output_value = value_map.get(return_id) orelse {
                std.debug.print("Error: Return value with ID {} not found\n", .{return_id});
                return error.ReturnValueNotFound;
            };

            const output_ptr = kernel_func.asValue().getParam(@intCast(num_inputs + i));
            // For now, assume output has same dtype as first input
            const output_dtype = try DType.fromString(config.external_inputs[0].tensor_info.?.dtype);
            const output_element_type = output_dtype.toLLVMType(self.context);
            const output_element_ptr = self.builder.buildGEP(output_element_type, output_ptr, &[_]*llvm.Value{global_idx}, "output.ptr");
            _ = self.builder.buildStore(output_value, output_element_ptr);
        }

        // Return from the kernel
        _ = self.builder.buildRetVoid();

        return kernel_func;
    }

    pub fn emitPTX(self: *KernelCodegen) ![]u8 {
        var error_msg: [*c]u8 = null;
        if (llvm.c.LLVMVerifyModule(@ptrCast(self.module), llvm.c.LLVMReturnStatusAction, &error_msg) != 0) {
            defer llvm.c.LLVMDisposeMessage(error_msg);
            std.debug.print("Module verification failed: {s}\n", .{error_msg});
            return error.ModuleVerificationFailed;
        }

        // Create NVPTX target
        var target: llvm.c.LLVMTargetRef = undefined;
        var target_error: [*c]u8 = null;
        if (llvm.c.LLVMGetTargetFromTriple(NVPTX_TARGET_TRIPLE, &target, &target_error) != 0) {
            defer llvm.c.LLVMDisposeMessage(target_error);
            std.debug.print("Failed to get target: {s}\n", .{target_error});
            return error.TargetNotFound;
        }

        // Create target machine
        const target_machine = llvm.c.LLVMCreateTargetMachine(
            target,
            NVPTX_TARGET_TRIPLE,
            NVPTX_CPU,
            NVPTX_FEATURES,
            llvm.c.LLVMCodeGenLevelDefault,
            llvm.c.LLVMRelocDefault,
            llvm.c.LLVMCodeModelDefault,
        );
        defer llvm.c.LLVMDisposeTargetMachine(target_machine);

        // Emit to memory buffer
        var mem_buf: llvm.c.LLVMMemoryBufferRef = undefined;
        var emit_error: [*c]u8 = null;
        if (llvm.c.LLVMTargetMachineEmitToMemoryBuffer(
            target_machine,
            @ptrCast(self.module),
            llvm.c.LLVMAssemblyFile, // PTX is assembly format
            &emit_error,
            &mem_buf,
        ) != 0) {
            defer llvm.c.LLVMDisposeMessage(emit_error);
            std.debug.print("Failed to emit PTX: {s}\n", .{emit_error});
            return error.PTXEmissionFailed;
        }
        defer llvm.c.LLVMDisposeMemoryBuffer(mem_buf);

        // Get PTX data from memory buffer
        const ptx_data = llvm.c.LLVMGetBufferStart(mem_buf);
        const ptx_size = llvm.c.LLVMGetBufferSize(mem_buf);

        // Copy PTX to result buffer and null-terminate for CUDA
        const result = try self.allocator.allocSentinel(u8, ptx_size, 0);
        @memcpy(result, ptx_data[0..ptx_size]);

        return result;
    }
};
