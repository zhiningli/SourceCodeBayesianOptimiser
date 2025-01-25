import torch
import re
from typing import str 

class ComponentExtractor:

    def __init__(self):
        r"""
        This class extract model and dataset specfications respectively from the 
        code string and instantiate each of them for further embedding purposes
        """

        self._model_string = None
        self.model_instance = None
        self._dataset_string = None
        self.datset_instance = None
        self.objective_func = None
        self._code_string = None

    @property
    def code_string(self):
        return self._code_string

    @code_string.setter
    def code_string(self, value: str):
        self._code_string = value

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