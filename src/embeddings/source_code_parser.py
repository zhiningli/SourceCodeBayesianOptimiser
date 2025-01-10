from src.mistral.mistral import MistralClient
from src.data.mistral_prompts.script_extraction_prompt import extract_information_from_script_prompts
import re

class Source_Code_Parser:

    def __init__(self):
        """
        Initializes the Source_Code_Parser class that can extract model, dataset respectively
        """

        self.mistral = MistralClient()

    def extract_information_from_code_string(self, code_str):
        """
        Extracts information from the given code string using the LLM.

        Parameters:
        code_str (str): The source code as a string.

        Returns:
        dict: A dictionary containing extracted information about the dataset, model, 
              hyperparameters, and evaluation metrics.
        """
        prompt = extract_information_from_script_prompts.format(
            source_code=code_str
        )
        source_code_information_response = self.mistral.call_codestral(prompt=prompt)

        return self.extract_model_and_dataset(source_code_information_response)

    def extract_model_and_dataset(self, source_code_information):
        # Extract the content inside <model> tags
        model_match = re.search(r"<model>(.*?)</model>", source_code_information, re.DOTALL)
        model_code = model_match.group(1).strip() if model_match else "No model found"
        
        # Extract the content inside <dataset> tags
        dataset_match = re.search(r"<dataset>(.*?)</dataset>", source_code_information, re.DOTALL)
        dataset_code = dataset_match.group(1).strip() if dataset_match else "No dataset found"
        
        # Return the extracted code as a dictionary or print them
        return {"model": model_code, "dataset": dataset_code}