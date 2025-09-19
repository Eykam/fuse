const std = @import("std");

pub const c = @cImport({
    @cInclude("llvm-c/Core.h");
    @cInclude("llvm-c/Target.h");
    @cInclude("llvm-c/TargetMachine.h");
    @cInclude("llvm-c/Analysis.h");
});

pub const Context = opaque {
    pub fn create() *Context {
        return @ptrCast(c.LLVMContextCreate());
    }

    pub fn destroy(self: *Context) void {
        c.LLVMContextDispose(@ptrCast(self));
    }
};

pub const Module = opaque {
    pub fn create(name: [*:0]const u8, context: *Context) *Module {
        return @ptrCast(c.LLVMModuleCreateWithNameInContext(name, @ptrCast(context)));
    }

    pub fn destroy(self: *Module) void {
        c.LLVMDisposeModule(@ptrCast(self));
    }

    pub fn printToString(self: *Module) [*:0]u8 {
        return c.LLVMPrintModuleToString(@ptrCast(self));
    }

    pub fn setDataLayout(self: *Module, layout: [*:0]const u8) void {
        c.LLVMSetDataLayout(@ptrCast(self), layout);
    }

    pub fn setTargetTriple(self: *Module, triple: [*:0]const u8) void {
        c.LLVMSetTarget(@ptrCast(self), triple);
    }
};

pub const Builder = opaque {
    pub fn create(context: *Context) *Builder {
        return @ptrCast(c.LLVMCreateBuilderInContext(@ptrCast(context)));
    }

    pub fn destroy(self: *Builder) void {
        c.LLVMDisposeBuilder(@ptrCast(self));
    }

    pub fn setInsertPoint(self: *Builder, block: *BasicBlock) void {
        c.LLVMPositionBuilderAtEnd(@ptrCast(self), @ptrCast(block));
    }

    pub fn buildRet(self: *Builder, value: ?*Value) *Value {
        if (value) |v| {
            return @ptrCast(c.LLVMBuildRet(@ptrCast(self), @ptrCast(v)));
        } else {
            return @ptrCast(c.LLVMBuildRetVoid(@ptrCast(self)));
        }
    }

    pub fn buildLoad(self: *Builder, ty: *Type, ptr: *Value, name: [*:0]const u8) *Value {
        return @ptrCast(c.LLVMBuildLoad2(@ptrCast(self), @ptrCast(ty), @ptrCast(ptr), name));
    }

    pub fn buildStore(self: *Builder, value: *Value, ptr: *Value) *Value {
        return @ptrCast(c.LLVMBuildStore(@ptrCast(self), @ptrCast(value), @ptrCast(ptr)));
    }

    pub fn buildGEP(self: *Builder, ty: *Type, ptr: *Value, indices: []const *Value, name: [*:0]const u8) *Value {
        if (indices.len == 0) {
            return @ptrCast(c.LLVMBuildGEP2(@ptrCast(self), @ptrCast(ty), @ptrCast(ptr), null, 0, name));
        } else {
            return @ptrCast(c.LLVMBuildGEP2(@ptrCast(self), @ptrCast(ty), @ptrCast(ptr), @ptrCast(@constCast(indices.ptr)), @intCast(indices.len), name));
        }
    }

    pub fn buildFCmpOGT(self: *Builder, lhs: *Value, rhs: *Value, name: [*:0]const u8) *Value {
        return @ptrCast(c.LLVMBuildFCmp(@ptrCast(self), c.LLVMRealOGT, @ptrCast(lhs), @ptrCast(rhs), name));
    }

    pub fn buildSelect(self: *Builder, cond: *Value, true_val: *Value, false_val: *Value, name: [*:0]const u8) *Value {
        return @ptrCast(c.LLVMBuildSelect(@ptrCast(self), @ptrCast(cond), @ptrCast(true_val), @ptrCast(false_val), name));
    }

    pub fn buildCall(self: *Builder, ty: *Type, fn_val: *Value, args: []const *Value, name: [*:0]const u8) *Value {
        if (args.len == 0) {
            return @ptrCast(c.LLVMBuildCall2(@ptrCast(self), @ptrCast(ty), @ptrCast(fn_val), null, 0, name));
        } else {
            return @ptrCast(c.LLVMBuildCall2(@ptrCast(self), @ptrCast(ty), @ptrCast(fn_val), @ptrCast(@constCast(args.ptr)), @intCast(args.len), name));
        }
    }

    pub fn buildAdd(self: *Builder, lhs: *Value, rhs: *Value, name: [*:0]const u8) *Value {
        return @ptrCast(c.LLVMBuildAdd(@ptrCast(self), @ptrCast(lhs), @ptrCast(rhs), name));
    }

    pub fn buildMul(self: *Builder, lhs: *Value, rhs: *Value, name: [*:0]const u8) *Value {
        return @ptrCast(c.LLVMBuildMul(@ptrCast(self), @ptrCast(lhs), @ptrCast(rhs), name));
    }

    pub fn buildFAdd(self: *Builder, lhs: *Value, rhs: *Value, name: [*:0]const u8) *Value {
        return @ptrCast(c.LLVMBuildFAdd(@ptrCast(self), @ptrCast(lhs), @ptrCast(rhs), name));
    }

    pub fn buildFSub(self: *Builder, lhs: *Value, rhs: *Value, name: [*:0]const u8) *Value {
        return @ptrCast(c.LLVMBuildFSub(@ptrCast(self), @ptrCast(lhs), @ptrCast(rhs), name));
    }

    pub fn buildFMul(self: *Builder, lhs: *Value, rhs: *Value, name: [*:0]const u8) *Value {
        return @ptrCast(c.LLVMBuildFMul(@ptrCast(self), @ptrCast(lhs), @ptrCast(rhs), name));
    }

    pub fn buildFDiv(self: *Builder, lhs: *Value, rhs: *Value, name: [*:0]const u8) *Value {
        return @ptrCast(c.LLVMBuildFDiv(@ptrCast(self), @ptrCast(lhs), @ptrCast(rhs), name));
    }

    pub fn buildICmpULT(self: *Builder, lhs: *Value, rhs: *Value, name: [*:0]const u8) *Value {
        return @ptrCast(c.LLVMBuildICmp(@ptrCast(self), c.LLVMIntULT, @ptrCast(lhs), @ptrCast(rhs), name));
    }

    pub fn buildCondBr(self: *Builder, cond: *Value, then_block: *BasicBlock, else_block: *BasicBlock) *Value {
        return @ptrCast(c.LLVMBuildCondBr(@ptrCast(self), @ptrCast(cond), @ptrCast(then_block), @ptrCast(else_block)));
    }

    pub fn buildRetVoid(self: *Builder) *Value {
        return @ptrCast(c.LLVMBuildRetVoid(@ptrCast(self)));
    }
};

pub const Type = opaque {
    pub fn voidType(context: *Context) *Type {
        return @ptrCast(c.LLVMVoidTypeInContext(@ptrCast(context)));
    }

    pub fn floatType(context: *Context) *Type {
        return @ptrCast(c.LLVMFloatTypeInContext(@ptrCast(context)));
    }

    pub fn doubleType(context: *Context) *Type {
        return @ptrCast(c.LLVMDoubleTypeInContext(@ptrCast(context)));
    }

    pub fn int8Type(context: *Context) *Type {
        return @ptrCast(c.LLVMInt8TypeInContext(@ptrCast(context)));
    }

    pub fn int16Type(context: *Context) *Type {
        return @ptrCast(c.LLVMInt16TypeInContext(@ptrCast(context)));
    }

    pub fn int32Type(context: *Context) *Type {
        return @ptrCast(c.LLVMInt32TypeInContext(@ptrCast(context)));
    }

    pub fn int64Type(context: *Context) *Type {
        return @ptrCast(c.LLVMInt64TypeInContext(@ptrCast(context)));
    }

    pub fn pointerType(element_type: *Type, address_space: u32) *Type {
        return @ptrCast(c.LLVMPointerType(@ptrCast(element_type), address_space));
    }

    pub fn functionType(return_type: *Type, param_types: []const *Type, is_var_arg: bool) *Type {
        return @ptrCast(c.LLVMFunctionType(@ptrCast(return_type), @ptrCast(@constCast(param_types.ptr)), @intCast(param_types.len), if (is_var_arg) 1 else 0));
    }
};

pub const Value = opaque {
    pub fn constFloat(ty: *Type, value: f64) *Value {
        return @ptrCast(c.LLVMConstReal(@ptrCast(ty), value));
    }

    pub fn constInt(ty: *Type, value: u64, sign_extend: bool) *Value {
        return @ptrCast(c.LLVMConstInt(@ptrCast(ty), value, if (sign_extend) 1 else 0));
    }

    pub fn setAlignment(self: *Value, alignment: u32) void {
        c.LLVMSetAlignment(@ptrCast(self), alignment);
    }

    pub fn setLinkage(self: *Value, linkage: c.LLVMLinkage) void {
        c.LLVMSetLinkage(@ptrCast(self), linkage);
    }

    pub fn getParam(self: *Value, index: u32) *Value {
        return @ptrCast(c.LLVMGetParam(@ptrCast(self), index));
    }

    pub fn addAttributeAtIndex(self: *Value, index: c_uint, attr: *Attribute) void {
        c.LLVMAddAttributeAtIndex(@ptrCast(self), index, @ptrCast(attr));
    }
};

pub const Function = opaque {
    pub fn addFunction(module: *Module, name: [*:0]const u8, ty: *Type) *Function {
        return @ptrCast(c.LLVMAddFunction(@ptrCast(module), name, @ptrCast(ty)));
    }

    pub fn appendBasicBlock(self: *Function, name: [*:0]const u8) *BasicBlock {
        return @ptrCast(c.LLVMAppendBasicBlock(@ptrCast(self), name));
    }

    pub fn asValue(self: *Function) *Value {
        return @ptrCast(self);
    }

    pub fn setLinkage(self: *Function, linkage: c.LLVMLinkage) void {
        c.LLVMSetLinkage(@ptrCast(self), linkage);
    }

    pub fn setCallingConv(self: *Function, conv: c.LLVMCallConv) void {
        c.LLVMSetFunctionCallConv(@ptrCast(self), conv);
    }
};

pub const BasicBlock = opaque {};

pub const Attribute = opaque {
    pub fn createString(context: *Context, key: [*:0]const u8, value: [*:0]const u8) *Attribute {
        return @ptrCast(c.LLVMCreateStringAttribute(@ptrCast(context), key, @intCast(std.mem.len(key)), value, @intCast(std.mem.len(value))));
    }
};

pub const AttributeFunctionIndex: c_uint = @bitCast(@as(c_int, -1));
pub const AttributeReturnIndex: c_uint = 0;

pub fn initializeNVPTXTarget() void {
    c.LLVMInitializeNVPTXTarget();
    c.LLVMInitializeNVPTXTargetInfo();
    c.LLVMInitializeNVPTXTargetMC();
    c.LLVMInitializeNVPTXAsmPrinter();
}
