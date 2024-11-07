source_code_validation_prompt = """

The aim of this task is to refine the source code provided below to address any errors.

Code:
<code>
{source_code}
</code> 

Error Message:
<error_msg>
{error_message}
</error_msg>

Instructions:
1. Amend the code to resolve any compilation or runtime errors based on the error message provided.
2. Pay particular attention to:
   - Dataset-specific preprocessing, as the source code may lack necessary adjustments.
   - Scaling of each input feature, ensuring consistency.

Format:
- Return the entire source code within triple backticks for Python (```python).
- Include the full code even if only minor edits were made.

Tone Context: You are a knowledgeable machine learning expert who understands both the dataset and the model in the source code. Your task is to correct the code for any errors, ensuring it is ready to run without issues.

"""

extract_information_from_source_code_prompt = """
Your task is to analyze and extract specific details from the provided source code.
<code>
{source_code}
</code> 
Please help me extract the following information:

1. "What dataset is being used in this code?"
2. "What model is being used in this code?"
3. "What are the hyperparameters for this model?"
4. "What evaluation metrics are present?"
5. "Does the dataset go through any scaling?"

Here is an example of how this should be done
```python
{{
    "dataset_name": "iris",
    "dataset_library": "sklearn",
    "model_type": "SVM",
    "hyperparameters": "{{'kernel': 'poly', 'C': 4.229101316230308, 'gamma': 'scale', 'coef0': 0.47320558504355414, 'random_state': 42}}",
    "evaluation_metrics": "accuracy",
    "dataset_scaling": "StandardScaler"
}}
```

Please return the information as a dictionary within triple backticks for Python, formatted as follows:

```python
{{
    "dataset_name": "<dataset name>",
    "dataset_library": "<dataset library>",
    "model_type": "<model name>",
    "hyperparameters": "<python set: hyperparameters>",
    "evaluation_metrics": "<evaluation metrics>",
    "dataset_scaling": "<scaling techniques>"
}}
"""


extract_dataset_from_source_code_prompt = """
Your task is to analyze the provided source code and create a new function that extracts and returns the training and test datasets.

<code>
{source_code}
</code>

Requirements for the new function:
1. **Library Imports**: Include all necessary library import statements at the top.
2. **Function Name**: The function should be named `extract_datasets`.
3. **Data Importing and Preprocessing**: Include all steps related to data loading, importing, and preprocessing from the original code. Ensure that any data transformations, scaling, encoding, or cleaning procedures are preserved.
4. **Return Statement**: Ensure that the function returns the training and test datasets as `pandas` DataFrames in the format:
   ```python
   return X_train, y_train, X_test, y_test
5. **Execution: Include a call to extract_datasets() at the end of the code so that the function is executed when the script runs.
Please ensure the code is enclosed in triple backticks and formatted as valid Python code. 
"""
