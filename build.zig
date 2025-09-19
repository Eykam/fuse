const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "cuda-codegen",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Find llvm-config
    const llvm_config_exe = b.findProgram(&.{
        "llvm-config-18",
        "llvm-config-17",
        "llvm-config-16",
        "llvm-config-15",
        "llvm-config",
    }, &.{}) catch {
        std.debug.print("Error: llvm-config not found. Please install LLVM development packages.\n", .{});
        std.debug.print("On Ubuntu/Debian: sudo apt-get install llvm-dev\n", .{});
        std.debug.print("On Fedora/RHEL: sudo dnf install llvm-devel\n", .{});
        std.debug.print("On macOS: brew install llvm\n", .{});
        std.process.exit(1);
    };

    // Get LLVM cflags
    const llvm_cflags = b.run(&.{ llvm_config_exe, "--cflags" });
    var cflags_iter = std.mem.tokenizeAny(u8, llvm_cflags, " \n\r\t");
    while (cflags_iter.next()) |flag| {
        if (std.mem.startsWith(u8, flag, "-I")) {
            exe.addIncludePath(.{ .cwd_relative = flag[2..] });
        } else if (std.mem.startsWith(u8, flag, "-D")) {
            const macro = flag[2..];
            if (std.mem.indexOf(u8, macro, "=")) |eq_pos| {
                exe.root_module.addCMacro(macro[0..eq_pos], macro[eq_pos + 1 ..]);
            } else {
                exe.root_module.addCMacro(macro, "");
            }
        }
    }

    // Get LLVM libs
    const llvm_libs = b.run(&.{ llvm_config_exe, "--libs" });
    var libs_iter = std.mem.tokenizeAny(u8, llvm_libs, " \n\r\t");
    while (libs_iter.next()) |lib| {
        if (std.mem.startsWith(u8, lib, "-l")) {
            exe.linkSystemLibrary(lib[2..]);
        }
    }

    // Get LLVM ldflags
    const llvm_ldflags = b.run(&.{ llvm_config_exe, "--ldflags" });
    var ldflags_iter = std.mem.tokenizeAny(u8, llvm_ldflags, " \n\r\t");
    while (ldflags_iter.next()) |flag| {
        if (std.mem.startsWith(u8, flag, "-L")) {
            exe.addLibraryPath(.{ .cwd_relative = flag[2..] });
        }
    }

    // Get LLVM system libs
    const llvm_system_libs = b.run(&.{ llvm_config_exe, "--system-libs" });
    var sys_libs_iter = std.mem.tokenizeAny(u8, llvm_system_libs, " \n\r\t");
    while (sys_libs_iter.next()) |lib| {
        if (std.mem.startsWith(u8, lib, "-l")) {
            exe.linkSystemLibrary(lib[2..]);
        }
    }

    exe.linkLibC();
    exe.linkLibCpp();

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Add CUDA runner executable
    const cuda_runner = b.addExecutable(.{
        .name = "cuda-runner",
        .root_source_file = b.path("src/cuda_runner.zig"),
        .target = target,
        .optimize = optimize,
    });

    cuda_runner.linkSystemLibrary("cuda");
    cuda_runner.linkLibC();

    b.installArtifact(cuda_runner);

    const run_cuda = b.addRunArtifact(cuda_runner);
    run_cuda.step.dependOn(b.getInstallStep());

    // Pass the PTX file as argument
    run_cuda.addArg("relu_kernel.ptx");

    const run_cuda_step = b.step("run-cuda", "Run the CUDA kernel");
    run_cuda_step.dependOn(&run_cuda.step);

    // Build shared library for Python FFI
    const ffi_lib = b.addSharedLibrary(.{
        .name = "kernel_fusion",
        .root_source_file = b.path("src/ffi.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Apply LLVM configuration to FFI library
    cflags_iter = std.mem.tokenizeAny(u8, llvm_cflags, " \n\r\t");
    while (cflags_iter.next()) |flag| {
        if (std.mem.startsWith(u8, flag, "-I")) {
            ffi_lib.addIncludePath(.{ .cwd_relative = flag[2..] });
        } else if (std.mem.startsWith(u8, flag, "-D")) {
            const macro = flag[2..];
            if (std.mem.indexOf(u8, macro, "=")) |eq_pos| {
                ffi_lib.root_module.addCMacro(macro[0..eq_pos], macro[eq_pos + 1 ..]);
            } else {
                ffi_lib.root_module.addCMacro(macro, "");
            }
        }
    }

    libs_iter = std.mem.tokenizeAny(u8, llvm_libs, " \n\r\t");
    while (libs_iter.next()) |lib| {
        if (std.mem.startsWith(u8, lib, "-l")) {
            ffi_lib.linkSystemLibrary(lib[2..]);
        }
    }

    ldflags_iter = std.mem.tokenizeAny(u8, llvm_ldflags, " \n\r\t");
    while (ldflags_iter.next()) |flag| {
        if (std.mem.startsWith(u8, flag, "-L")) {
            ffi_lib.addLibraryPath(.{ .cwd_relative = flag[2..] });
        }
    }

    sys_libs_iter = std.mem.tokenizeAny(u8, llvm_system_libs, " \n\r\t");
    while (sys_libs_iter.next()) |lib| {
        if (std.mem.startsWith(u8, lib, "-l")) {
            ffi_lib.linkSystemLibrary(lib[2..]);
        }
    }

    ffi_lib.linkLibC();
    ffi_lib.linkLibCpp();
    ffi_lib.linkSystemLibrary("cuda");  // Link CUDA driver library

    b.installArtifact(ffi_lib);

    const ffi_step = b.step("ffi", "Build the shared library for Python FFI");
    ffi_step.dependOn(b.getInstallStep());

    // Tests
    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Apply same LLVM configuration to tests
    cflags_iter = std.mem.tokenizeAny(u8, llvm_cflags, " \n\r\t");
    while (cflags_iter.next()) |flag| {
        if (std.mem.startsWith(u8, flag, "-I")) {
            unit_tests.addIncludePath(.{ .cwd_relative = flag[2..] });
        } else if (std.mem.startsWith(u8, flag, "-D")) {
            const macro = flag[2..];
            if (std.mem.indexOf(u8, macro, "=")) |eq_pos| {
                unit_tests.root_module.addCMacro(macro[0..eq_pos], macro[eq_pos + 1 ..]);
            } else {
                unit_tests.root_module.addCMacro(macro, "");
            }
        }
    }

    libs_iter = std.mem.tokenizeAny(u8, llvm_libs, " \n\r\t");
    while (libs_iter.next()) |lib| {
        if (std.mem.startsWith(u8, lib, "-l")) {
            unit_tests.linkSystemLibrary(lib[2..]);
        }
    }

    ldflags_iter = std.mem.tokenizeAny(u8, llvm_ldflags, " \n\r\t");
    while (ldflags_iter.next()) |flag| {
        if (std.mem.startsWith(u8, flag, "-L")) {
            unit_tests.addLibraryPath(.{ .cwd_relative = flag[2..] });
        }
    }

    sys_libs_iter = std.mem.tokenizeAny(u8, llvm_system_libs, " \n\r\t");
    while (sys_libs_iter.next()) |lib| {
        if (std.mem.startsWith(u8, lib, "-l")) {
            unit_tests.linkSystemLibrary(lib[2..]);
        }
    }

    unit_tests.linkLibC();
    unit_tests.linkLibCpp();

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}