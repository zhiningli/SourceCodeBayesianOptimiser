# Import necessary modules
from src.mistral.mistral import MistralClient  # Custom client for interacting with the LLM
from src.data.mistral_prompts.data_validation_prompt import extract_information_from_source_code_prompt, extract_dataset_from_source_code_prompt  # Custom prompt for extracting info
import numpy as np
import pandas as pd

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
        self.dataset_statistics = {
            "number_of_features": None,
            "number_of_samples": None,
            "feature_to_sample_ratio": None,
            "feature_scaling": False,
            "dataset_name": None,
            "dataset_library": None,
            "linearity_score": None,
        }

        self.model_statistics = {
            "model_type": None,
            "model_hyperparameters": {}
        }

        self.evaluation_metrics = {}

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
            source_code_information_response = self.mistral.call_codestral(prompt=prompt)

            source_code_information = self.mistral.extract_code_block(source_code_information_response)

            self.datasets = None
            self.dataset_statistics["dataset_name"] = source_code_information["dataset_name"]
            self.dataset_statistics["dataset_library"] = source_code_information["dataset_library"]
            self.dataset_statistics["feature_scaling"] = source_code_information["dataset_scaling"] 

            self.model_statistics["model_type"] = source_code_information["model_type"]
            self.model_statistics["model_hyperparameters"] = source_code_information["hyperparameters"]
            
            self.evaluation_metrics.add(source_code_information["evaluation_metrics"])

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

    def extract_dataset_from_code_string(self, code_str):
        try:
            prompt = extract_dataset_from_source_code_prompt.format(source_code=code_str)

            response = self.mistral.call_codestral(prompt=prompt)

            dataset_extraction_code = self.mistral.extract_code_block(response)

            try:
                namespace = {}
                exec(dataset_extraction_code, namespace)
                self.datasets = namespace["extract_datasets()"]()
                return True
            except Exception as e:
                print("Error executing code from mistral", e)
                return False

    
        except AttributeError as e:
            print(f"Error: Attribute issue - {e}")
            return {"error": "Atrribute issue during mistral interaction"}

        except TypeError as e:
            print(f"Error: Type issue - {e}")
            return {"error": "Type issue in input or method call."}
    
        except Exception as e:
            print(f"An unexpected error occured: {e}")
            return {"error": "An unexpected error occured during processing"}
        
    def perform_statistical_analysis(self):
        if self.datasets:
            X_train, y_train, X_test, y_test = self.datasets
            
            # Calculate linearity score
            self.dataset_statistics["linearity_score"] = self._calculate_aggregate_linearity_score(X_train, y_train)
            
            # Extract the number of features and samples using the shape attribute
            self.dataset_statistics["number_of_features"] = X_train.shape[1]  # Number of columns (features)
            self.dataset_statistics["number_of_samples"] = X_train.shape[0]   # Number of rows (samples)
            
            # Calculate the feature-to-sample ratio
            self.dataset_statistics["feature_to_sample_ratio"] = (
                self.dataset_statistics["number_of_features"] / self.dataset_statistics["number_of_samples"]
            )


    def _calculate_aggregate_linearity_score(X, y, threshold=0.5):
        correlation_matrix = X.corrwith(y, method='pearson')

        linear_features_count = (correlation_matrix.abs() > threshold).sum()

        total_features = len(correlation_matrix)
        linearity_score = linear_features_count / total_features
        return linearity_score
        
