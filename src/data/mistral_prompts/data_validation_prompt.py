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
5. "What are the ultimate training dataset (X_train, y_train) and test dataset(X_test, y_test) used for fitting and evaluating the model?

Please return the information as a dictionary within triple backticks for Python, formatted as follows:

```python
{
    "dataset": "<dataset information>",
    "model": "<model information>",
    "hyperparameters": "<hyperparameters>",
    "evaluation_metrics": "<evaluation metrics>"
}
"""