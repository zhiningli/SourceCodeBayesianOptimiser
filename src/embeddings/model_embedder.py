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
        print("Extracted architecture: ", architecture)

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
            self.generic_visit(node)
            self.in_init = False
        elif node.name == "forward":
            self.in_forward = True
            self.generic_visit(node)
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
                        self.architecture["layers"][layer_name] = {
                            "type": layer_type,
                            "attributes": {},
                            "children": {}
                        }

                        # Extract attributes for standard layers like Linear
                        if layer_type == "Linear":
                            args = node.value.args
                            if len(args) >= 2:
                                self.architecture["layers"][layer_name]["attributes"] = {
                                    "in_features": self._get_argument_value(args[0]),
                                    "out_features": self._get_argument_value(args[1])
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
        Extract the value of an argument. Handles constants and variable references.

        Args:
            arg: AST node representing the argument.

        Returns:
            The value of the argument if constant, or its name if it's a variable reference.
        """
        if isinstance(arg, ast.Constant):  # Python 3.8+
            return arg.value
        elif isinstance(arg, ast.Name):
            return arg.id
        elif isinstance(arg, ast.Attribute):
            return f"{arg.value.id}.{arg.attr}"
        else:
            return None

    def visit_Expr(self, node):
        if self.in_forward and isinstance(node.value, ast.Call):
            if isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name):
                layer_call = f"{node.value.func.value}.{node.value.func.attr}()"
                self.architecture["forward_flow"].append(layer_call)



        