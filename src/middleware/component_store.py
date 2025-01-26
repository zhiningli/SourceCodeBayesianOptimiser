import torch
import re
from typing import Callable
import inspect

# A centralised managed store for code string, model string, dataset string as where as their respective instances.
# All instances are stored in a seperate namespace so it does not interfere with other part of the module 
# For further enhancement, consider using a more isolated environment such as pysandbox for running these instances

class ComponentStore:

    def __init__(self):
        r"""
        This class extract model and dataset specfications respectively from the 
        code string and instantiate each of them for further embedding purposes
        """
        self._code_string: str = None
        self._model_string: str = None
        self._dataset_string: str = None

        self.model_instance: torch.nn.Module = None
        self.dataset_instance: torch.utils.data.Dataset = None
        self.objective_func: Callable = None

        self.namespace = {}

    @property
    def code_string(self) -> str:
        return self._code_string

    @code_string.setter
    def code_string(self, value: str) -> None:
        self._code_string = value

    @property
    def model_string(self) -> str:
        return self._model_string

    @model_string.setter
    def model_string(self, value: str) -> None:
        self._model_string = value
    
    @property
    def dataset_string(self) -> str:
        return self._dataset_string
    
    @dataset_string.setter
    def dataset_string(self, value: str) -> None:
        self._dataset_string = value
    
    def validate_code_string(self, target_string: str) -> None:
        r"""
        Validate the code string and check that it does not contain sensitive keywords that may potentially harm the module.
        """
        forbidden_patterns = [r"exec", r"os\.", r"sys\.", r"subprocess"]
        for pattern in forbidden_patterns:
            if re.search(pattern, target_string):
                raise ValueError(f"Forbidden pattern found in code: {pattern}")

    def instantiate_code_classes(self) -> None:

        self.validate_code_string(target_string=self.code_string)
        exec(self.code_string, self.namespace)

        # Extracting objective function
        self.objective_func = self.namespace.get("train_simple_nn")
        if not callable(self.objective_func):
            raise ValueError("No valid objective function named 'train_simple_nn' detechted")
        print("Extracted Objective Fucntion: train_simple_nn")

        # Extracting the model class
        self.model_instance = self.namespace.get("Model")
        if not self.model_instance or not issubclass(self.model_instance, torch.nn.Module):
            raise ValueError("No valid model class named 'Model' found")
        print(f"Extracted Model: {self.model_instance}")

        # Extracting the dataset object
        self.dataset_instance = self.namespace.get("train_dataset")
        if not self.dataset_instance or not isinstance(self.dataset_instance, torch.utils.data.Dataset):
            raise ValueError("No valid dataset named 'train_dataset' found")
        print(f"Extracted dataset: {self.dataset_instance}")


        # # Adding some utility functions to the namespace for code analysis
        # self.namespace["inspect_model_class"] = self._inspect_model_class
    
    # def _inspect_model_class(self) -> dict:
    #     """
    #     Inspect the model class to analyze its structure.

    #     Returns:
    #         dict: A structured dictionary representing the class details
    #     """

    #     if not self.model_instance:
    #         raise ValueError("Model class is not available for inspection")
        
    #     model_class = self.model_instance
    #     architecture = {}

    #     # Analyze __init__ and forward methods
    #     try:
    #         init_source = inspect.getsource(model_class.__init__)
    #         architecture["init_method"] = init_source
    #     except OSError:
    #         architecture["init_method"] = "Could not retrieve __init__ source"

    #     try:
    #         forward_source = inspect.getsource(model_class.forward)
    #         architecture["forward_method"] = forward_source
    #     except OSError:
    #         architecture["forward_method"] = "Could not retrieve forward source"

    #     return architecture
    

            