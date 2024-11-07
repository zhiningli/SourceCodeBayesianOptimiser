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

Please return the information as a dictionary within triple backticks for Python, formatted as follows:

```python
{
    "dataset_name": "<dataset name>",
    "dataset_library": "<dataset library>"
    "model_type": "<model name>",
    "hyperparameters": "<python set: hyperparameters>",
    "evaluation_metrics": "<evaluation metrics>"
}
"""

extract_dataset_from_source_code_prompt = """
Your task is to analyze the provided source code and generate a new code string that extracts and returns the training and test datasets.

<code>
{source_code}
</code>

Requirements for the new code string:
1. **Data Importing and Preprocessing**: Include all steps related to data loading, importing, and preprocessing as present in the original code. Ensure that any data transformations, scaling, encoding, or cleaning procedures are preserved.
2. **Return Statement**: Ensure that the code returns the training and test datasets as `pandas` DataFrames (`X_train`, `y_train`, `X_test`, `y_test`).

Format the response within triple backticks for Python:

```python
# Include the data importing, loading, and preprocessing steps from the original code
# Ensure the final lines return the datasets as DataFrames
return (X_train, y_train), (X_test, y_test)
"""
