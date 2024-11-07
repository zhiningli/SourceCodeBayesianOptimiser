# Import necessary modules
from src.mistral.mistral import MistralClient  # Custom client for interacting with the LLM
from src.data.mistral_prompts.data_validation_prompt import extract_information_from_source_code_prompt  # Custom prompt for extracting info

class CodeStrAnalyser:
    """
    A class to analyze a code string and extract specific information using the MistralClient.
    This class formats and sends a prompt to an LLM and returns extracted information.
    """

    def __init__(self):
        """
        Initializes the CodeStrAnalyser class by creating an instance of MistralClient.
        """
        # Initialize the MistralClient instance for LLM interaction
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
        try:
            # Format the prompt with the given source code
            prompt = extract_information_from_source_code_prompt.format(
                source_code=code_str
            )

            # Call the LLM through MistralClient and get the response
            source_code_information = self.mistral.call_codestral(prompt=prompt)

            # Return the extracted information in dictionary format
            return source_code_information

        except AttributeError as e:
            # Handle potential attribute errors, such as an improperly initialized MistralClient
            print(f"Error: Attribute issue - {e}")
            return {"error": "Attribute issue during LLM interaction."}

        except TypeError as e:
            # Handle issues with the input type (e.g., non-string input for code_str)
            print(f"Error: Type issue - {e}")
            return {"error": "Type issue in input or method call."}

        except Exception as e:
            # Catch any other general exceptions and log them
            print(f"An unexpected error occurred: {e}")
            return {"error": "An unexpected error occurred during processing."}

    def 