from src.mistral.mistral import MistralClient
from src.data.mistral_prompts.script_extraction_prompt import extract_information_from_script_prompts

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

        source_code_information = self.mistral.extract_code_block(source_code_information_response)

        self.extract_model_and_dataset(source_code_information)

        return True

    def extract_model_and_dataset(self, source_code_information):
        print("response fro mistral: ", source_code_information)