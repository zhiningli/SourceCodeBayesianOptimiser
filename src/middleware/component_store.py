import torch
import re
from typing import Optional

class ComponentStore:

    def __init__(self):
        r"""
        This class extract model and dataset specfications respectively from the 
        code string and instantiate each of them for further embedding purposes
        """

        self._model_string: str | None = None
        self.model_instance: str | None = None
        self._dataset_string = str | None= None
        self.dataset_instance = None
        self.objective_func = None
        self._code_string = None
        self.namespace = {}

    @property
    def code_string(self):
        return self._code_string

    @code_string.setter
    def code_string(self, value: str):
        self._code_string = value

    @property
    def model_string(self):
        return self._model_string

    @model_string.setter
    def model_string(self, value: str):
        self._model_string = value
    
    @property
    def dataset_string(self):
        return self._dataset_string
    
    @dataset_string.setter
    def dataset_string(self, value: str):
        self._dataset_string = value

    def extract_architecture(self):
        """
        Extract the architecture of the instantiated PyTorch model
        """
    
    def validate_code_string(self, target_string: str):

        forbidden_patterns = [r"exec", r"os\.", r"sys\.", r"subprocess"]
        for pattern in forbidden_patterns:
            if re.search(pattern, target_string):
                raise ValueError(f"Forbidden pattern found in code: {pattern}")

    def instantiate_code_classes(self):

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
    

    

            