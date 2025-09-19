# CUDA Codegen TODO

## ✅ Completed
- [x] AST parser with 3-pass alias resolution
- [x] Function call mapping and parameter extraction
- [x] Variable aliasing across function boundaries

## 🔄 Next Steps

### 1. Enhanced Parameter Metadata
- [ ] Add dtype/rank/shape information to config entries
- [ ] Extract tensor metadata from function arguments
- [ ] Support for different data types (float32, int32, etc.)
- [ ] Handle dynamic vs static shapes

### 2. Variable Modification Detection
- [ ] **CRITICAL**: Handle variables modified between kernel calls
- [ ] Example: `x = kernel1(); x = x + 1; y = kernel2(x)` - can't fuse!
- [ ] Track variable assignments and mutations
- [ ] Break fusion chains when variables are modified in Python
- [ ] Detect in-place operations that invalidate fusion
- [ ] Create separate fusion groups for modified variables

### 3. Python Decorator Interface
- [ ] Create `@fuse` decorator to mark entry points
- [ ] Decorator should capture function signature and metadata
- [ ] Auto-detect kernel functions within decorated scope
- [ ] Generate unique cache keys from function + arguments

### 4. Python-Zig FFI Integration
- [ ] Create FFI interface to pass config from Python to Zig
- [ ] Serialize kernel config to format Zig can consume
- [ ] Handle memory management between Python/Zig boundary
- [ ] Return compiled kernel names/handles back to Python

### 5. Zig Kernel Fusion Engine
- [ ] Implement dependency graph builder in Zig
- [ ] Create fusion validator (compatibility checking)
- [ ] Implement kernel fusion logic/optimizer
- [ ] LLVM IR kernel codegen with fused operations
- [ ] PTX generation and compilation

### 6. Kernel Execution Caching System
- [ ] Cache based on function signature + argument metadata
- [ ] Hash table for cache lookup (args -> kernel_name)
- [ ] Cache miss: trigger full kernel extraction → fusion → codegen
- [ ] Cache hit: directly execute cached kernel via FFI
- [ ] Cache invalidation strategy

### 7. End-to-End Integration
- [ ] Complete Python → Zig → CUDA pipeline
- [ ] Error handling and validation throughout pipeline
- [ ] Performance benchmarking vs separate kernel execution
- [ ] Memory management and cleanup

## 🎯 Target Workflow

```python
@fuse
def my_computation(a: Tensor[float32], b: Tensor[float32]) -> Tensor[float32]:
    temp1 = ops.add(a, b)        # Fusable
    temp2 = ops.relu(temp1)      # Fusable with above

    # Python modification breaks fusion chain
    temp2 = temp2 * 2            # NOT fusable - Python operation

    result = ops.multiply(temp2, a)  # New fusion group starts here
    return result

# First call: extract → detect modifications → create fusion groups → compile → cache
result1 = my_computation(x, y)
```

## 🚨 Critical Cases to Handle

```python
# Case 1: Variable reassignment
x = ops.add(a, b)
x = x + 1              # Python op - breaks fusion
y = ops.relu(x)        # Can't fuse with first ops.add

# Case 2: In-place modification
x = ops.add(a, b)
x += 1                 # In-place - breaks fusion
y = ops.relu(x)

# Case 3: Multiple references
x = ops.add(a, b)
temp = x * 2           # x still valid for fusion
y = ops.relu(x)        # Can still fuse with ops.add
z = ops.multiply(temp, y)  # temp can't be fused
```

## 📋 Implementation Priority
1. **Variable modification detection** (critical for correctness)
2. Parameter metadata extraction
3. Decorator interface design
4. FFI boundary establishment
5. Basic Zig fusion engine
6. Caching system integration
7. End-to-End testing and optimization