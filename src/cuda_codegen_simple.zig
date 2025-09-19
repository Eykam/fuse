const std = @import("std");
const llvm = @import("llvm_c.zig");

pub const CudaCodegen = struct {
    allocator: std.mem.Allocator,
    context: *llvm.Context,
    module: *llvm.Module,
    builder: *llvm.Builder,
    kernel_name: []const u8,

    pub fn init(allocator: std.mem.Allocator, kernel_name: []const u8) !CudaCodegen {
        llvm.initializeNVPTXTarget();

        const context = llvm.Context.create();
        const module_name = try std.fmt.allocPrintZ(allocator, "{s}_module", .{kernel_name});
        defer allocator.free(module_name);

        const module = llvm.Module.create(module_name, context);

        // Set NVPTX target
        module.setDataLayout("e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64");
        module.setTargetTriple("nvptx64-nvidia-cuda");

        const builder = llvm.Builder.create(context);

        return CudaCodegen{
            .allocator = allocator,
            .context = context,
            .module = module,
            .builder = builder,
            .kernel_name = kernel_name,
        };
    }

    pub fn deinit(self: *CudaCodegen) void {
        self.builder.destroy();
        self.module.destroy();
        self.context.destroy();
    }

    pub fn generateParallelReLUKernel(self: *CudaCodegen) !void {
        // Create function signature: void relu_kernel(float* input, float* output, int n)
        const void_type = llvm.Type.voidType(self.context);
        const float_type = llvm.Type.floatType(self.context);
        const int32_type = llvm.Type.int32Type(self.context);
        const float_ptr_type = llvm.Type.pointerType(float_type, 1); // address space 1 for global memory

        const param_types = [_]*llvm.Type{ float_ptr_type, float_ptr_type, int32_type };
        const func_type = llvm.Type.functionType(void_type, &param_types, false);

        const kernel_name_z = try self.allocator.dupeZ(u8, self.kernel_name);
        defer self.allocator.free(kernel_name_z);
        const kernel_func = llvm.Function.addFunction(self.module, kernel_name_z, func_type);

        // Set external linkage
        kernel_func.asValue().setLinkage(llvm.c.LLVMExternalLinkage);

        // Set calling convention to PTX kernel (71 = LLVMPTXKernelCallConv)
        llvm.c.LLVMSetFunctionCallConv(@ptrCast(kernel_func.asValue()), 71);

        // Mark as CUDA kernel
        const nvvm_kernel_attr = llvm.Attribute.createString(self.context, "nvvm.kernel", "1");
        kernel_func.asValue().addAttributeAtIndex(llvm.AttributeFunctionIndex, nvvm_kernel_attr);

        // Create entry block
        const entry_block = kernel_func.appendBasicBlock("entry");
        self.builder.setInsertPoint(entry_block);

        // Get function parameters
        const input_ptr = kernel_func.asValue().getParam(0);
        const output_ptr = kernel_func.asValue().getParam(1);
        const n = kernel_func.asValue().getParam(2);

        // Get CUDA built-in functions for thread indexing using our fixed wrapper functions
        const tid_x_func = self.getThreadIdXFunc();
        const block_dim_x_func = self.getBlockDimXFunc();
        const block_idx_x_func = self.getBlockIdxXFunc();

        // Call the intrinsic functions to get actual thread indices
        const empty_args = [_]*llvm.Value{};
        const tid_x = self.builder.buildCall(int32_type, tid_x_func, empty_args[0..], "tid.x");
        const block_dim_x = self.builder.buildCall(int32_type, block_dim_x_func, empty_args[0..], "blockDim.x");
        const block_idx_x = self.builder.buildCall(int32_type, block_idx_x_func, empty_args[0..], "blockIdx.x");

        // block_offset = blockIdx.x * blockDim.x
        const block_offset = @as(*llvm.Value, @ptrCast(llvm.c.LLVMBuildMul(
            @ptrCast(self.builder),
            @ptrCast(block_idx_x),
            @ptrCast(block_dim_x),
            "block.offset"
        )));

        // global_idx = blockIdx.x * blockDim.x + threadIdx.x
        const global_idx = @as(*llvm.Value, @ptrCast(llvm.c.LLVMBuildAdd(
            @ptrCast(self.builder),
            @ptrCast(block_offset),
            @ptrCast(tid_x),
            "global.idx"
        )));

        // Check bounds: if (global_idx < n)
        const bounds_cmp = @as(*llvm.Value, @ptrCast(llvm.c.LLVMBuildICmp(
            @ptrCast(self.builder),
            llvm.c.LLVMIntSLT,
            @ptrCast(global_idx),
            @ptrCast(n),
            "bounds.check"
        )));

        // Create blocks for bounds check
        const in_bounds_block = kernel_func.appendBasicBlock("in_bounds");
        const exit_block = kernel_func.appendBasicBlock("exit");

        _ = @as(*llvm.Value, @ptrCast(llvm.c.LLVMBuildCondBr(
            @ptrCast(self.builder),
            @ptrCast(bounds_cmp),
            @ptrCast(in_bounds_block),
            @ptrCast(exit_block)
        )));

        // In bounds block: compute ReLU
        self.builder.setInsertPoint(in_bounds_block);

        // Load input[global_idx]
        const input_gep = self.builder.buildGEP(float_type, input_ptr, &[_]*llvm.Value{global_idx}, "input.ptr");
        const input_val = self.builder.buildLoad(float_type, input_gep, "input.val");

        // ReLU: max(0, input)
        const zero_float = llvm.Value.constFloat(float_type, 0.0);
        const gt = self.builder.buildFCmpOGT(input_val, zero_float, "gt");
        const relu_val = self.builder.buildSelect(gt, input_val, zero_float, "relu.val");

        // Store to output[global_idx]
        const output_gep = self.builder.buildGEP(float_type, output_ptr, &[_]*llvm.Value{global_idx}, "output.ptr");
        _ = self.builder.buildStore(relu_val, output_gep);

        // Jump to exit
        _ = @as(*llvm.Value, @ptrCast(llvm.c.LLVMBuildBr(@ptrCast(self.builder), @ptrCast(exit_block))));

        // Exit block: return
        self.builder.setInsertPoint(exit_block);
        _ = self.builder.buildRet(null);
    }

    fn getThreadIdXFunc(self: *CudaCodegen) *llvm.Value {
        const existing = llvm.c.LLVMGetNamedFunction(@ptrCast(self.module), "llvm.nvvm.read.ptx.sreg.tid.x");
        if (existing) |func| {
            return @ptrCast(func);
        }

        const int32_type = llvm.Type.int32Type(self.context);
        const func_type = llvm.Type.functionType(int32_type, &[_]*llvm.Type{}, false);
        return @ptrCast(llvm.c.LLVMAddFunction(@ptrCast(self.module), "llvm.nvvm.read.ptx.sreg.tid.x", @ptrCast(func_type)));
    }

    fn getBlockDimXFunc(self: *CudaCodegen) *llvm.Value {
        const existing = llvm.c.LLVMGetNamedFunction(@ptrCast(self.module), "llvm.nvvm.read.ptx.sreg.ntid.x");
        if (existing) |func| {
            return @ptrCast(func);
        }

        const int32_type = llvm.Type.int32Type(self.context);
        const func_type = llvm.Type.functionType(int32_type, &[_]*llvm.Type{}, false);
        return @ptrCast(llvm.c.LLVMAddFunction(@ptrCast(self.module), "llvm.nvvm.read.ptx.sreg.ntid.x", @ptrCast(func_type)));
    }

    fn getBlockIdxXFunc(self: *CudaCodegen) *llvm.Value {
        const existing = llvm.c.LLVMGetNamedFunction(@ptrCast(self.module), "llvm.nvvm.read.ptx.sreg.ctaid.x");
        if (existing) |func| {
            return @ptrCast(func);
        }

        const int32_type = llvm.Type.int32Type(self.context);
        const func_type = llvm.Type.functionType(int32_type, &[_]*llvm.Type{}, false);
        return @ptrCast(llvm.c.LLVMAddFunction(@ptrCast(self.module), "llvm.nvvm.read.ptx.sreg.ctaid.x", @ptrCast(func_type)));
    }

    pub fn emitPTX(self: *CudaCodegen) ![]u8 {
        // Verify module
        var error_msg: [*c]u8 = null;
        if (llvm.c.LLVMVerifyModule(@ptrCast(self.module), llvm.c.LLVMReturnStatusAction, &error_msg) != 0) {
            defer llvm.c.LLVMDisposeMessage(error_msg);
            std.debug.print("Module verification failed: {s}\n", .{error_msg});
            return error.ModuleVerificationFailed;
        }

        // Initialize NVPTX target
        llvm.c.LLVMInitializeNVPTXTargetInfo();
        llvm.c.LLVMInitializeNVPTXTarget();
        llvm.c.LLVMInitializeNVPTXTargetMC();
        llvm.c.LLVMInitializeNVPTXAsmPrinter();

        // Get the NVPTX target
        var target: llvm.c.LLVMTargetRef = null;
        var error_message: [*c]u8 = null;
        if (llvm.c.LLVMGetTargetFromTriple("nvptx64-nvidia-cuda", &target, &error_message) != 0) {
            defer llvm.c.LLVMDisposeMessage(error_message);
            std.debug.print("Failed to get NVPTX target: {s}\n", .{error_message});
            return error.TargetNotFound;
        }

        // Create target machine
        const target_machine = llvm.c.LLVMCreateTargetMachine(
            target,
            "nvptx64-nvidia-cuda",
            "sm_50", // GPU compute capability
            "",      // Features
            llvm.c.LLVMCodeGenLevelDefault,
            llvm.c.LLVMRelocDefault,
            llvm.c.LLVMCodeModelDefault,
        );
        defer llvm.c.LLVMDisposeTargetMachine(target_machine);

        // Generate PTX assembly
        var output_memory_buffer: llvm.c.LLVMMemoryBufferRef = null;
        var error_msg2: [*c]u8 = null;
        if (llvm.c.LLVMTargetMachineEmitToMemoryBuffer(
            target_machine,
            @ptrCast(self.module),
            llvm.c.LLVMAssemblyFile,
            &error_msg2,
            &output_memory_buffer,
        ) != 0) {
            defer llvm.c.LLVMDisposeMessage(error_msg2);
            std.debug.print("Failed to emit PTX: {s}\n", .{error_msg2});
            return error.CodeGenFailed;
        }
        defer llvm.c.LLVMDisposeMemoryBuffer(output_memory_buffer);

        // Get PTX string
        const ptx_ptr = llvm.c.LLVMGetBufferStart(output_memory_buffer);
        const ptx_size = llvm.c.LLVMGetBufferSize(output_memory_buffer);

        const result = try self.allocator.alloc(u8, ptx_size);
        @memcpy(result, ptx_ptr[0..ptx_size]);

        return result;
    }
};