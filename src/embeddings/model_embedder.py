from transformers import RobertaTokenizer, RobertaModel
import torch
import ast
from abc import ABC, abstractclassmethod


class Model_Embedder(ABC):
    
    def __init__(self):
        self.tokeniser = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")   
    
    def embed_code_snippet(self, code_snippet: str) -> torch.Tensor:
        inputs = self.tokeniser(code_snippet, 
                                return_tensors="pt", 
                                truncation=True, 
                                padding=True, 
                                max_length=128)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)

        return embeddings
    
    @abstractclassmethod
    def embed_source_code(self, code_snippet: str) -> torch.Tensor:
        pass


class Direct_Model_Code_Embedder(Model_Embedder):

    def embed_source_code(self, code_snippet: str) -> torch.Tensor:
        return self.embed_code_snippet(code_snippet=code_snippet)


class Model_Architecture_Code_Embedder(Model_Embedder):

    def __init__(self, class_name = "Model"):
        super().__init__()
        self.extractor = ModelArchitectureExtractor(class_name)

    def embed_source_code(self, code_string: str) -> torch.Tensor:
        """
        Embed the architecture of a model class via AST and convert it into a vector representation.
        
        Args:
            code_string: str
            The python source code string contraining the model class
            
        Returns:
            torch.Tensor: The embedded representation of the model's architecture.
        """

        tree = ast.parse(code_string)
        self.extractor.set_tree(tree)
        self.extractor.visit(tree)
        architecture = self.extractor.architecture

        return self.embed_code_snippet(code_snippet=str(architecture))
        


class ModelArchitectureExtractor(ast.NodeVisitor):
    """
    Extract the architecture of a PyTorch model class from its AST with optimized single traversal.
    """

    def __init__(self, class_name: str):
        self.class_name = class_name
        self.tree = None  
        self.architecture = {"layers": {}, "forward_flow": []}
        self.class_definitions = {}  
        self.in_init = False
        self.in_forward = False
        self.layer_count = 0

    def set_tree(self, tree: ast.AST):
        """
        Assign the AST tree to the extractor and collect class definitions.
        
        Args:
            tree (ast.AST): The parsed AST tree.
        """
        self.tree = tree
        self._collect_class_definitions(tree)

    def _collect_class_definitions(self, tree: ast.AST):
        """
        Collect all class definitions in the AST for later use.

        Args:
            tree (ast.AST): The parsed AST tree.
        """
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self.class_definitions[node.name] = node

    def visit_ClassDef(self, node):
        if node.name == self.class_name:
            self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if node.name == "__init__":
            self.in_init = True
            self.generic_visit(node)  # Traverse the entire __init__ method
            self.in_init = False
        elif node.name == "forward":
            self.in_forward = True
            self.generic_visit(node)  # Traverse the entire forward method
            self.in_forward = False

    def visit_Assign(self, node):
        if self.in_init:
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                    layer_name = target.attr
                    self.layer_count += 1

                    # Check if the layer is a standard layer or a submodule
                    if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                        layer_type = node.value.func.attr
                        layer_attributes = self._extract_layer_attributes(
                            layer_type, node.value.args, node.value.keywords
                        )
                        self.architecture["layers"][layer_name] = {
                            "type": layer_type,
                            "attributes": layer_attributes,
                            "children": {}
                        }

                    # Handle submodules
                    elif isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                        custom_module_name = node.value.func.id
                        if custom_module_name in self.class_definitions:
                            nested_node = self.class_definitions[custom_module_name]
                            nested_architecture = self._extract_nested_module(nested_node)
                            self.architecture["layers"][layer_name] = {
                                "type": custom_module_name,
                                "attributes": {},
                                "children": nested_architecture
                            }

        if self.in_forward:
            if isinstance(node.value, ast.Call): 
                if isinstance(node.value.func, ast.Attribute):
                    func_name = self._resolve_func_name(node.value.func)
                    args = [self._get_argument_value(arg) for arg in node.value.args]
                    call_repr = f"{func_name}({', '.join(map(str, args))})"
                    self.architecture["forward_flow"].append(call_repr)

                elif isinstance(node.value.func, ast.Name):
                    func_call = f"{node.value.func.id}({', '.join(self._get_argument_value(arg) for arg in node.value.args)})"
                    self.architecture["forward_flow"].append(func_call)

    def _resolve_func_name(self, func_node):
        """
        Resolve the name of a function or method call, including attributes.

        Args:
            func_node (ast.Attribute): The AST node for the function or method.

        Returns:
            str: The resolved name as a string.
        """
        if isinstance(func_node, ast.Attribute):
            # Resolve chained attributes like self.conv1 or x.size
            value = self._get_argument_value(func_node.value)
            return f"{value}.{func_node.attr}"
        elif isinstance(func_node, ast.Name):
            # For standalone function names
            return func_node.id
        else:
            return "unknown_func"


    def _extract_layer_attributes(self, layer_type: str, args: list, keywords: list) -> dict:
        attributes = {}

        if layer_type == "Linear":
            attributes["in_features"] = None
            attributes["out_features"] = None

            # Handle positional arguments
            if len(args) >= 2:
                attributes["in_features"] = self._get_argument_value(args[0])
                attributes["out_features"] = self._get_argument_value(args[1])

            # Handle keyword arguments
            for kw in keywords:
                if kw.arg == "in_features":
                    attributes["in_features"] = self._get_argument_value(kw.value)
                elif kw.arg == "out_features":
                    attributes["out_features"] = self._get_argument_value(kw.value)

        elif layer_type in ["Conv1d", "Conv2d"]:
            attributes.update({
                "in_channels": None,
                "out_channels": None,
                "kernel_size": None,
                "stride": 1,  # Default stride
                "padding": 0  # Default padding
            })

            # Positional arguments
            if len(args) >= 2:
                attributes["in_channels"] = self._get_argument_value(args[0])
                attributes["out_channels"] = self._get_argument_value(args[1])
            if len(args) >= 3:
                attributes["kernel_size"] = self._get_argument_value(args[2])

            # Keyword arguments
            for kw in keywords:
                attributes[kw.arg] = self._get_argument_value(kw.value)

        return attributes



    def _extract_nested_module(self, class_node: ast.ClassDef) -> dict:
        """
        Extract the architecture of a nested module.

        Args:
            class_node (ast.ClassDef): The AST node for the class.

        Returns:
            dict: The architecture of the nested module.
        """
        nested_extractor = ModelArchitectureExtractor(class_node.name)
        nested_extractor.set_tree(self.tree)
        nested_extractor.visit(class_node)
        return nested_extractor.architecture

    def _get_argument_value(self, arg):
        """
        Extract the value of an argument. Handles constants, variable references, attributes, 
        and complex expressions such as binary operations, function calls, and subscripts.

        Args:
            arg: AST node representing the argument.

        Returns:
            str: The resolved value of the argument as a string.
        """
        if isinstance(arg, ast.Constant):  # Handle literal values (Python 3.8+)
            return repr(arg.value)  # Ensure strings are quoted
        elif isinstance(arg, ast.Name):  # Handle variable references (e.g., `x`)
            return arg.id
        elif isinstance(arg, ast.Attribute):  # Handle attributes (e.g., `self.size`)
            base = self._get_argument_value(arg.value)
            return f"{base}.{arg.attr}" if base else f"{arg.attr}"
        elif isinstance(arg, ast.BinOp):  # Handle binary operations (e.g., 32 * (input_size // 2))
            left = self._get_argument_value(arg.left)
            right = self._get_argument_value(arg.right)
            operator = self._get_operator(arg.op)
            return f"({left} {operator} {right})"
        elif isinstance(arg, ast.Call):  # Handle function calls (e.g., `size(0)`)
            func_name = self._get_argument_value(arg.func)
            args = [self._get_argument_value(a) for a in arg.args]
            kwargs = [f"{kw.arg}={self._get_argument_value(kw.value)}" for kw in arg.keywords]
            all_args = ", ".join(args + kwargs)
            return f"{func_name}({all_args})"
        elif isinstance(arg, ast.Subscript):  # Handle subscript (e.g., `x[0]`)
            value = self._get_argument_value(arg.value)
            slice = self._get_argument_value(arg.slice)
            return f"{value}[{slice}]"
        elif isinstance(arg, ast.UnaryOp):  # Handle unary operations (e.g., `-x`)
            operand = self._get_argument_value(arg.operand)
            operator = self._get_operator(arg.op)
            return f"{operator}{operand}"
        elif isinstance(arg, ast.Compare):  # Handle comparisons (e.g., `input_size > 0`)
            left = self._get_argument_value(arg.left)
            comparators = [self._get_argument_value(comp) for comp in arg.comparators]
            operators = [self._get_operator(op) for op in arg.ops]
            return f"{left} {' '.join(op + ' ' + comp for op, comp in zip(operators, comparators))}"
        elif isinstance(arg, ast.List):  # Handle lists (e.g., `[1, 2, 3]`)
            elements = [self._get_argument_value(e) for e in arg.elts]
            return f"[{', '.join(elements)}]"
        elif isinstance(arg, ast.Tuple):  # Handle tuples (e.g., `(1, 2)`)
            elements = [self._get_argument_value(e) for e in arg.elts]
            return f"({', '.join(elements)})"
        elif isinstance(arg, ast.Dict):  # Handle dictionaries (e.g., `{"a": 1, "b": 2}`)
            keys = [self._get_argument_value(k) for k in arg.keys]
            values = [self._get_argument_value(v) for v in arg.values]
            return f"{{{', '.join(f'{k}: {v}' for k, v in zip(keys, values))}}}"
        else:
            return "unknown"  # Return a placeholder for unsupported or unrecognized types


    def _get_operator(self, op):
        """
        Map AST operator nodes to their string representations.

        Args:
            op: AST operator node.

        Returns:
            str: The string representation of the operator.
        """
        operators = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
            ast.FloorDiv: "//",
            ast.Pow: "**",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.And: "and",
            ast.Or: "or",
            ast.BitAnd: "&",
            ast.BitOr: "|",
            ast.BitXor: "^",
            ast.USub: "-",
            ast.UAdd: "+",
        }
        return operators.get(type(op), "?")



        