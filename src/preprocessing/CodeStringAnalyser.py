from src.mistral.mistral import MistralClient
from src.data.mistral_prompts.data_evluation_prompt import (
    extract_information_from_source_code_prompt, 
    extract_dataset_from_source_code_prompt, 
    extract_information_and_datasets_prompt
)
import numpy as np
import pandas as pd
import json

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
            "dataset_name": None,
            "dataset_library": None,
            "number_of_features": None,
            "number_of_samples": None,
            "feature_to_sample_ratio": None,
            "feature_scaling": False,
            "linearity_score": None,
            "test_size": None,
            "number_of_classes": None,
            "cross_validation": None,
            "feature_selection": None,
            "imputation": None,
            "encoding": None,
        }

        self.model_statistics = {
            "model_type": None,
            "model_source": None,
            "objective": None,
            "model_hyperparameters": {},
        }

        self.evaluation_metrics = None
        self.datasets = None

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
            prompt = extract_information_from_source_code_prompt.format(
                source_code=code_str
            )
            # Call the LLM through MistralClient and get the response
            source_code_information_response = self.mistral.call_codestral(prompt=prompt)

            # Extract code block from response
            source_code_information = self.mistral.extract_code_block(source_code_information_response)

            # Ensure source_code_information is parsed correctly (assuming it's JSON-like)
            if isinstance(source_code_information, str):
                source_code_information = json.loads(source_code_information)

            # Safeguard key access and update internal state only if keys are present
            self.dataset_statistics.update({
                "dataset_name": source_code_information.get("dataset_name", "Unknown"),
                "dataset_library": source_code_information.get("dataset_library", "Unknown"),
                "feature_scaling": source_code_information.get("dataset_scaling", False),
                "test_size": source_code_information.get("test_size", None),
                "number_of_classes": source_code_information.get("number_of_classes", None),
                "cross_validation": source_code_information.get("cross_validation", None),
                "feature_selection": source_code_information.get("feature_selection", None),
                "imputation": source_code_information.get("imputation", None),
                "encoding": source_code_information.get("encoding", None),
            })

            self.model_statistics.update({
                "model_type": source_code_information.get("model_type", "Unknown"),
                "model_source": source_code_information.get("model_source", "Unknown"),
                "objective": source_code_information.get("objective", "Unknown"),
                "model_hyperparameters": source_code_information.get("hyperparameters", {}),
            })

            self.evaluation_metrics = source_code_information.get("evaluation_metrics", None)

            return True

        except (AttributeError, KeyError, ValueError) as e:
            print(f"Error: {e}")
            return {"error": str(e)}

        except Exception as e:
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
                self.datasets = namespace["extract_datasets"]()

                X_train = pd.DataFrame(self.datasets[0])
                X_test = pd.DataFrame(self.datasets[2])
                
                y_train = self._convert_to_series(self.datasets[1])
                y_test = self._convert_to_series(self.datasets[3])
                
                self.datasets = (X_train, y_train, X_test, y_test)
                return True
            except Exception as e:
                print("Error executing code from mistral:", e)
                return False

        except AttributeError as e:
            print(f"Error: Attribute issue - {e}")
            return {"error": "Attribute issue during mistral interaction"}

        except TypeError as e:
            print(f"Error: Type issue - {e}")
            return {"error": "Type issue in input or method call."}
            
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {"error": "An unexpected error occurred during processing"}

    def extract_information_and_dataset_from_code_str(self, code_str):
        """
        Extracts both information and dataset from the code string.
        """
        prompt = extract_information_and_datasets_prompt.format(source_code=code_str)
        response = self.mistral.call_codestral(prompt=prompt)
        source_code_information = self.mistral.extract_code_block(response)
        if isinstance(source_code_information, str):

            source_code_information = json.loads(source_code_information)

        self.dataset_statistics.update({
            "dataset_name": source_code_information.get("dataset_name", "Unknown"),
            "dataset_library": source_code_information.get("dataset_library", "Unknown"),
            "feature_scaling": source_code_information.get("dataset_scaling", False),
            "test_size": source_code_information.get("test_size", None),
            "number_of_classes": source_code_information.get("number_of_classes", None),
            "cross_validation": source_code_information.get("cross_validation", None),
            "feature_selection": source_code_information.get("feature_selection", None),
            "imputation": source_code_information.get("imputation", None),
            "encoding": source_code_information.get("encoding", None),
        })

        self.model_statistics.update({
            "model_type": source_code_information.get("model_type", "Unknown"),
            "model_source": source_code_information.get("model_source", "Unknown"),
            "objective": source_code_information.get("objective", "Unknown"),
            "model_hyperparameters": source_code_information.get("hyperparameters", {}),
        })

        self.evaluation_metrics = source_code_information.get("evaluation_metrics", None)
        dataset_extraction_code = source_code_information.get("extract_datasets_function", None)


        try:
            namespace = {}
            exec(dataset_extraction_code, namespace)
            self.datasets = namespace["extract_datasets"]()

            X_train = pd.DataFrame(self.datasets[0])
            X_test = pd.DataFrame(self.datasets[2])

            y_train = self._convert_to_series(self.datasets[1])
            y_test = self._convert_to_series(self.datasets[3])

            self.datasets = (X_train, y_train, X_test, y_test)
            return True
        except Exception as e:
            print("Error executing code from mistral:", e)
            return False


    def _convert_to_series(self, data):
        """
        Converts the input data to a 1-dimensional pandas Series.
        Handles cases where the input is a DataFrame with shape (n, 1) or a numpy array.
        """
        if isinstance(data, pd.DataFrame) and data.shape[1] == 1:
            return pd.Series(data.squeeze())
        elif isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 1:
            return pd.Series(data.squeeze())
        else:
            return pd.Series(data)

    def perform_statistical_analysis(self):
        if self.datasets:
            X_train, y_train, X_test, y_test = self.datasets

            self.dataset_statistics["linearity_score"] = self._calculate_aggregate_linearity_score(X_train, y_train)

            self.dataset_statistics["number_of_features"] = X_train.shape[1]
            self.dataset_statistics["number_of_samples"] = X_train.shape[0]

            self.dataset_statistics["feature_to_sample_ratio"] = (
                self.dataset_statistics["number_of_features"] / self.dataset_statistics["number_of_samples"]
            )
        else:
            raise ValueError("Dataset not loaded for the code string analyser")

    def _calculate_aggregate_linearity_score(self, X, y, threshold=0.5):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X should be a pandas DataFrame.")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y should be a pandas Series or a 1-dimensional numpy array.")

        correlation_matrix = X.corrwith(y, method='pearson')
        linear_features_count = (correlation_matrix.abs() > threshold).sum()

        total_features = len(correlation_matrix)
        linearity_score = linear_features_count / total_features

        return linearity_score
