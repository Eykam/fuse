"""
Python AST parser to extract kernel calls from Python code.
Identifies calls to ops.* methods and generates a config with unique IDs.
"""

import ast
import json
import inspect
import symtable
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict


@dataclass
class KernelCall:
    """Represents a single kernel call"""

    id: int  # Unique ID for this call
    op: str  # Operation type (add, subtract, etc.)
    input_ids: List[int]  # IDs of inputs (negative for external inputs)
    output_id: int  # ID of the output (same as id for simplicity)
    lineno: int  # Line number in source


@dataclass
class TensorInfo:
    """Tensor metadata information"""

    dtype: str  # "float32", "int32", "float64", etc.
    rank: int  # Number of dimensions (0=scalar, 1=vector, 2=matrix, etc.)
    shape: Optional[List[Optional[int]]] = (
        None  # [1024, 512] or [None, 512] for dynamic
    )


@dataclass
class ExternalInput:
    """External input variable (function parameters)"""

    id: int  # Negative ID to distinguish from kernel outputs
    name: str  # Variable name/parameter name
    tensor_info: Optional[TensorInfo] = None  # Tensor metadata
    size: Optional[int] = None  # Buffer size in bytes


@dataclass
class ExternalOutput:
    """External output specification"""

    id: int  # Output ID
    size: int  # Buffer size in bytes


@dataclass
class KernelConfig:
    """Configuration extracted from Python code"""

    function_name: str
    kernel_calls: List[KernelCall]
    external_inputs: List[ExternalInput]
    external_outputs: List[Dict] = None  # Will be populated at runtime
    return_ids: List[int] = None  # IDs of values that are returned


class KernelExtractor(ast.NodeVisitor):
    """
    Extract kernel calls from Python AST.

    How it works:
    1. Walks through Python AST looking for ops.* calls
    2. Assigns unique IDs to each kernel call output
    3. Tracks data flow by mapping variable names to IDs
    4. External inputs (function parameters) get negative IDs

    Example:
        def foo(a, b):          # a gets ID -1, b gets ID -2
            temp = ops.add(a, b) # add gets ID 1, inputs are [-1, -2]
            result = ops.relu(temp) # relu gets ID 2, input is [1]
            return result        # return_ids = [2]
    """

    def __init__(self, function_name: str = ""):
        self.function_name = function_name

        # Pass 1: Function call mappings
        self.call_map: Dict[Tuple[str, str], List[str]] = {}

        # Pass 2: Function parameters
        self.function_params: Dict[str, List[str]] = {}

        # Pass 3: Alias mappings and kernel extraction
        self.aliases: Dict[str, str] = {}  # callee_param -> caller_var
        self.kernel_calls: List[KernelCall] = []
        self.external_inputs: Dict[str, tuple] = (
            {}
        )  # scoped_name -> (param_id, name)
        self.var_to_id: Dict[str, int] = {}  # scoped_name -> kernel_output_id
        self.return_ids: List[int] = []

        # @fuse function detection
        self.fuse_functions: Set[str] = (
            set()
        )  # Scoped names of @fuse functions

        # Global ID counters
        self.next_call_id = 1
        self.next_external_id = -1

        # Current processing state
        self.scope: List[str] = []
        self.current_function: Optional[str] = None

        # Valid kernel operations
        self.kernel_ops = {"add", "subtract", "multiply", "divide", "relu"}

    def extract_from_source(self, source_code: str) -> KernelConfig:
        """Main entry point: run 3-pass analysis"""
        tree = ast.parse(source_code)

        # Pass 1: Extract function call mappings
        print("=== Pass 1: Extracting function calls ===")
        self._pass1_extract_calls(tree)
        print(f"Found {len(self.call_map)} function calls:")
        for (caller, callee), args in self.call_map.items():
            print(f"  {caller} -> {callee}({', '.join(args)})")

        # Pass 2: Extract function parameters using symtable
        print("\n=== Pass 2: Extracting function parameters ===")
        self._pass2_extract_parameters(source_code)
        print(f"Found parameters for {len(self.function_params)} functions:")
        for func_name, params in self.function_params.items():
            print(f"  {func_name}: {params}")

        # Step 3: Create alias mappings (data processing)
        print("\n=== Step 3: Creating alias mappings ===")
        self._create_alias_mappings()
        print(f"Created {len(self.aliases)} aliases:")
        for callee_param, caller_var in self.aliases.items():
            print(f"  {callee_param} -> {caller_var}")

        # Pass 3: Extract kernels with alias resolution
        print("\n=== Pass 3: Extracting kernels ===")
        self.visit(tree)

        print(f"\nFound {len(self.fuse_functions)} @fuse functions:")
        for func_name in self.fuse_functions:
            print(f"  {func_name}")

        return self.get_config()

    def _pass1_extract_calls(self, tree: ast.AST):
        """Pass 1: Extract function call mappings"""

        class CallExtractor(ast.NodeVisitor):
            def __init__(self, parent):
                self.parent = parent
                self.current_function = (
                    "main"  # Default to main for module-level calls
                )

            def visit_FunctionDef(self, node):
                if node.name.startswith("__"):
                    return
                old_func = self.current_function
                self.current_function = node.name
                self.generic_visit(node)
                self.current_function = old_func

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id.startswith(
                    "pattern"
                ):
                    callee = node.func.id
                    arguments = []
                    for arg in node.args:
                        if isinstance(arg, ast.Name):
                            arguments.append(arg.id)

                    if self.current_function:
                        key = (self.current_function, callee)
                        self.parent.call_map[key] = arguments

                self.generic_visit(node)

        extractor = CallExtractor(self)
        extractor.visit(tree)

    def _pass2_extract_parameters(self, source_code: str):
        """Pass 2: Extract function parameters using symtable"""
        try:
            st = symtable.symtable(source_code, "filename", "exec")
            self._process_symbol_table(st)
        except Exception as e:
            print(f"Error in symtable analysis: {e}")

    def _process_symbol_table(self, symbol_table):
        """Process symbol table to find function parameters"""
        if symbol_table.get_type() == "function":
            func_name = symbol_table.get_name()
            parameters = []
            for identifier in symbol_table.get_identifiers():
                symbol = symbol_table.lookup(identifier)
                if symbol.is_parameter():
                    parameters.append(identifier)
            if parameters:
                self.function_params[func_name] = parameters

        for child in symbol_table.get_children():
            self._process_symbol_table(child)

    def _create_alias_mappings(self):
        """Step 3: Create alias mappings from call map and function parameters"""
        for (caller, callee), arguments in self.call_map.items():
            if callee not in self.function_params:
                continue

            params = self.function_params[callee]
            for i, arg_var in enumerate(arguments):
                if i < len(params):
                    param_name = params[i]
                    caller_var = f"{caller}::{arg_var}"
                    callee_param = f"{callee}::{param_name}"
                    self.aliases[callee_param] = caller_var

    def get_scoped_name(self, var_name: str) -> str:
        """Build scoped name using current scope stack"""
        if self.scope:
            return "::".join(self.scope) + "::" + var_name
        return var_name

    def get_or_create_external_input(self, var_name: str) -> int:
        """
        Get ID for an external input using scoped variable name.
        Variables with same scope path share the same ID.
        """
        scoped_name = self.get_scoped_name(var_name)
        if scoped_name not in self.external_inputs:
            param_id = self.next_external_id
            self.next_external_id -= 1
            self.external_inputs[scoped_name] = (param_id, var_name)
        return self.external_inputs[scoped_name][0]

    def get_or_create_external_input_by_name(self, scoped_name: str) -> int:
        """
        Get ID for an external input by its scoped name (used for alias resolution).
        """
        if scoped_name not in self.external_inputs:
            param_id = self.next_external_id
            self.next_external_id -= 1
            # Extract just the variable name from scoped name
            var_name = (
                scoped_name.split("::")[-1]
                if "::" in scoped_name
                else scoped_name
            )
            self.external_inputs[scoped_name] = (param_id, var_name)
        return self.external_inputs[scoped_name][0]

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Enter function scope and check for @fuse decorator"""
        self.scope.append(node.name)

        # Check if this function has a @fuse decorator
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "fuse":
                # Build scoped name for this @fuse function
                # Use scope without the current function name since we just added it
                if len(self.scope) > 1:
                    # Nested function - exclude current function name
                    parent_scope = "::".join(self.scope[:-1])
                    scoped_name = f"{parent_scope}::{node.name}"
                else:
                    # Top-level function
                    scoped_name = node.name
                self.fuse_functions.add(scoped_name)
                break
            elif (
                isinstance(decorator, ast.Attribute)
                and decorator.attr == "fuse"
            ):
                # Handle cases like @module.fuse
                if len(self.scope) > 1:
                    parent_scope = "::".join(self.scope[:-1])
                    scoped_name = f"{parent_scope}::{node.name}"
                else:
                    scoped_name = node.name
                self.fuse_functions.add(scoped_name)
                break

        self.generic_visit(node)
        self.scope.pop()

    def visit_Call(self, node: ast.Call):
        """
        Don't process calls here - they're handled in visit_Assign or get_value_id.
        This prevents duplicate processing of kernel calls.
        """
        pass

    def visit_Assign(self, node: ast.Assign):
        """
        Process assignment statements like: temp = ops.add(a, b)
        Also detects variable modifications that break fusion chains.
        """
        # Only handle simple assignments (not tuple unpacking)
        if len(node.targets) != 1:
            self.generic_visit(node)  # Continue visiting children
            return

        target = node.targets[0]
        if not isinstance(target, ast.Name):
            self.generic_visit(node)
            return

        output_var = target.id  # Variable being assigned to
        scoped_var = self.get_scoped_name(output_var)

        # Validate that the assignment value is allowed in fusion functions
        self._validate_assignment_value(node.value, node.lineno)

        # Check if the right side is a kernel call
        if isinstance(node.value, ast.Call):
            call_id = self.process_call(node.value, node.lineno)
            if call_id is not None:
                # This is a kernel call assignment
                # Record that this variable now holds this kernel's output
                self.var_to_id[scoped_var] = call_id
        elif isinstance(node.value, ast.Name):
            # Simple variable assignment (e.g., temp = result) is allowed
            pass

        # Don't call generic_visit here - we've already processed the assignment

    def visit_AugAssign(self, node: ast.AugAssign):
        """
        Handle in-place assignments like: x += 1, x *= 2
        These are not allowed in fusion functions.
        """
        if isinstance(node.target, ast.Name):
            var_name = node.target.id
            raise RuntimeError(
                f"In-place assignment '{var_name} {self._get_aug_op_symbol(node.op)}= ...' "
                f"at line {node.lineno} is not allowed in fusion functions. "
                f"Only ops.* kernel operations are permitted."
            )

        self.generic_visit(node)

    def _get_aug_op_symbol(self, op: ast.operator) -> str:
        """Get the symbol for an augmented assignment operator."""
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
            ast.Pow: "**",
            ast.LShift: "<<",
            ast.RShift: ">>",
            ast.BitOr: "|",
            ast.BitXor: "^",
            ast.BitAnd: "&",
            ast.FloorDiv: "//",
        }
        return op_map.get(type(op), "?")

    def _validate_assignment_value(self, value: ast.AST, lineno: int):
        """
        Validate that assignment values are only kernel operations.
        Throws RuntimeError if non-kernel operations are detected.
        """
        if isinstance(value, ast.Call):
            # Check if it's a kernel call (but don't process it - just validate)
            is_kernel = self._is_kernel_call(value)
            if not is_kernel:
                # Not a kernel call - check if it's a @fuse function
                if self._is_fuse_function_call(value):
                    call_str = self._ast_to_string(value)
                    raise RuntimeError(
                        f"Nested @fuse function call '{call_str}' at line {lineno} "
                        f"is not allowed. Please move @fuse to the outermost function only."
                    )
                else:
                    # Generic non-kernel call
                    call_str = self._ast_to_string(value)
                    raise RuntimeError(
                        f"Non-kernel function call '{call_str}' at line {lineno} "
                        f"is not allowed in fusion functions. Only ops.* kernel operations are permitted."
                    )
        elif isinstance(value, ast.BinOp):
            # Binary operations like x + 1, y * 2 are not allowed
            op_str = self._ast_to_string(value)
            raise RuntimeError(
                f"Python binary operation '{op_str}' at line {lineno} "
                f"is not allowed in fusion functions. Only ops.* kernel operations are permitted."
            )
        elif isinstance(value, ast.UnaryOp):
            # Unary operations like -x, +x are not allowed
            op_str = self._ast_to_string(value)
            raise RuntimeError(
                f"Python unary operation '{op_str}' at line {lineno} "
                f"is not allowed in fusion functions. Only ops.* kernel operations are permitted."
            )
        elif isinstance(value, ast.Name):
            # Simple variable assignment is okay (e.g., temp = result)
            pass
        else:
            # Any other complex expression is not allowed
            expr_str = self._ast_to_string(value)
            raise RuntimeError(
                f"Complex expression '{expr_str}' at line {lineno} "
                f"is not allowed in fusion functions. Only ops.* kernel operations and simple variable assignments are permitted."
            )

    def _is_kernel_call(self, call: ast.Call) -> bool:
        """
        Check if a call is a kernel operation (ops.*) without processing it.
        """
        if isinstance(call.func, ast.Attribute):
            if (
                isinstance(call.func.value, ast.Name)
                and call.func.value.id == "ops"
            ):
                kernel_name = call.func.attr
                return kernel_name in self.kernel_ops
        return False

    def _is_fuse_function_call(self, call: ast.Call) -> bool:
        """
        Check if a function call is to a @fuse decorated function.
        """
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
            # Check if this function name is in our @fuse registry
            # Try both simple name and scoped name
            if func_name in self.fuse_functions:
                return True
            # Also check with current scope prefix
            scoped_name = self.get_scoped_name(func_name)
            return scoped_name in self.fuse_functions
        return False

    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert AST node to string representation for error messages."""
        try:
            import astor

            return astor.to_source(node).strip()
        except ImportError:
            # Fallback if astor is not available
            return f"<expression at line {getattr(node, 'lineno', '?')}>"

    def process_call(self, call: ast.Call, lineno: int) -> Optional[int]:
        """
        Check if a function call is a kernel operation (ops.*).
        Returns the output ID if it's a kernel, None otherwise.
        """
        # Check for pattern: ops.add(...), ops.relu(...), etc.
        if isinstance(call.func, ast.Attribute):
            if (
                isinstance(call.func.value, ast.Name)
                and call.func.value.id == "ops"
            ):
                kernel_name = call.func.attr
                if kernel_name in self.kernel_ops:
                    return self.create_kernel_call(
                        kernel_name, call.args, lineno
                    )
        return None

    def create_kernel_call(
        self, kernel_name: str, args: List[ast.AST], lineno: int
    ) -> int:
        """
        Create a kernel call record and return its output ID.
        """
        # Process each argument to get its ID
        input_ids = []
        for arg in args:
            arg_id = self.get_value_id(arg, lineno)
            if arg_id is not None:
                input_ids.append(arg_id)

        # Assign unique ID to this kernel call
        call_id = self.next_call_id
        self.next_call_id += 1

        # Create the kernel call record
        kernel_call = KernelCall(
            id=call_id,
            op=kernel_name,
            input_ids=input_ids,
            output_id=call_id,  # Using same ID for simplicity
            lineno=lineno,
        )
        self.kernel_calls.append(kernel_call)

        return call_id

    def get_value_id(self, node: ast.AST, lineno: int) -> Optional[int]:
        """
        Get the ID for a value with alias resolution.
        - A variable holding a previous kernel's output (positive ID)
        - An external input/parameter (negative ID, with alias resolution)
        - A nested kernel call (process it and return its ID)
        """
        if isinstance(node, ast.Name):
            var_name = node.id
            scoped_var = self.get_scoped_name(var_name)

            # Check if this variable holds a kernel output
            if scoped_var in self.var_to_id:
                return self.var_to_id[scoped_var]

            # Check for alias resolution
            if scoped_var in self.aliases:
                # This parameter is aliased to a caller variable
                caller_var = self.aliases[scoped_var]
                return self.get_or_create_external_input_by_name(caller_var)
            else:
                # Regular external input
                return self.get_or_create_external_input(var_name)

        elif isinstance(node, ast.Call):
            # Nested call like: ops.relu(ops.add(a, b))
            return self.process_call(node, lineno)

        return None

    def visit_Return(self, node: ast.Return):
        """
        Track what values are returned from the function.
        """
        if node.value is None:
            return

        if isinstance(node.value, ast.Name):
            # Single return value
            scoped_var = self.get_scoped_name(node.value.id)
            if scoped_var in self.var_to_id:
                self.return_ids.append(self.var_to_id[scoped_var])
            else:
                # This might be a parameter being returned directly
                # Check if it's in the current function's parameter list
                if self.scope:  # We're inside a function
                    current_func_name = self.scope[-1]  # Last item in scope is current function
                    if node.value.id in self.function_params.get(current_func_name, []):
                        # It's a parameter of the current function
                        param_id = self.get_or_create_external_input(node.value.id)
                        self.return_ids.append(param_id)

        elif isinstance(node.value, ast.Call):
            # Return value is a function call - process it and get its ID
            # TODO: This is a temporary fix. AST parser should be reworked to process ops.* calls in visit_Call
            value_id = self.get_value_id(node.value, node.lineno)
            if value_id is not None:
                self.return_ids.append(value_id)

        elif isinstance(node.value, ast.Tuple):
            # Multiple return values
            for elt in node.value.elts:
                if isinstance(elt, ast.Name):
                    scoped_var = self.get_scoped_name(elt.id)
                    if scoped_var in self.var_to_id:
                        self.return_ids.append(self.var_to_id[scoped_var])
                    elif scoped_var in self.external_inputs:
                        # Returning a parameter directly without any operations
                        param_id, _ = self.external_inputs[scoped_var]
                        self.return_ids.append(param_id)
                elif isinstance(elt, ast.Call):
                    # Return value is a function call - process it and get its ID
                    # TODO: This is a temporary fix. AST parser should be reworked to process ops.* calls in visit_Call
                    value_id = self.get_value_id(elt, node.lineno)
                    if value_id is not None:
                        self.return_ids.append(value_id)

        # Don't call generic_visit here - we've already processed the return value

    def get_config(self) -> KernelConfig:
        """Build the final configuration"""
        external_inputs = [
            ExternalInput(id=param_id, name=scoped_name)
            for scoped_name, (param_id, name) in self.external_inputs.items()
        ]

        # Sort for consistency
        external_inputs.sort(key=lambda x: x.id, reverse=True)

        return KernelConfig(
            function_name=self.function_name,
            kernel_calls=self.kernel_calls,
            external_inputs=external_inputs,
            external_outputs=[],  # Will be populated at runtime
            return_ids=self.return_ids,
        )


def config_to_json(config: KernelConfig) -> str:
    """Convert config to JSON for passing to Zig"""
    return json.dumps(asdict(config), indent=2)


def analyze_patterns_main():
    """Analyze patterns.py with integrated 3-pass approach"""
    import patterns

    # Get the source of the entire patterns module
    print("=== Analyzing patterns.py with integrated approach ===")

    source = inspect.getsource(patterns)
    extractor = KernelExtractor(function_name="patterns_main")
    config = extractor.extract_from_source(source)

    print(f"Total kernels found: {len(config.kernel_calls)}")
    print(f"External inputs: {[inp.name for inp in config.external_inputs]}")
    print(f"Returns: {config.return_ids}")
    print("\nFull config:")
    print(config_to_json(config))

    return config


if __name__ == "__main__":
    analyze_patterns_main()
