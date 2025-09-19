# Fuse

A lightweight JIT compiler for GPU tensor operations that automatically fuses operations into optimized CUDA kernels.

## Overview

Fuse is a simple just-in-time (JIT) compilation framework that takes tensor operation definitions and automatically fuses them into efficient CUDA kernels. It generates PTX (Parallel Thread Execution) code that runs directly on NVIDIA GPUs, eliminating the overhead of multiple kernel launches.

## Features

- **Operation Fusion**: Automatically combines multiple tensor operations into a single kernel
- **JIT Compilation**: Compiles operations to PTX at runtime for maximum flexibility
- **PTX Generation**: Produces optimized PTX code for direct GPU execution
- **FFI Support**: Provides foreign function interface bindings for shared library integration
- **Minimal Overhead**: Lightweight design focused on performance

## How It Works

1. Define your tensor operations using the Fuse API
2. Fuse automatically analyzes operation dependencies and fusion opportunities
3. Generates optimized PTX code combining multiple operations
4. Executes the fused kernel on GPU with minimal overhead

## Build

```bash
# Build instructions here
```

## Usage

```bash
# Usage examples here
```

## Requirements

- Zig 0.14
- NVIDIA GPU with CUDA support
- CUDA Toolkit
- LLVM ≥15

## License

MIT