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

extract_information_and_datasets_prompt = """
Your task is to analyze the provided source code to firstly: extract key details and secondly: create a function to return the training and test datasets.

<code>
{source_code}
</code>

Please help me extract the following information from the source code:

1. "What dataset is being used in this code?"
2. "What model is being used in this code?"
3. "What are the hyperparameters for this model?"
4. "What evaluation metrics are present?"
5. "Does the dataset go through any scaling?"

For the second task, please adhere to the following requirements for the source code extracted:

Please provide the function code in Python format within the dictionary field `extract_datasets_function`, ensuring it is properly indented and formatted as valid Python code.
Library Imports: Include all necessary library import statements at the top.
Data Importing and Preprocessing: Include all steps related to data loading, importing, and preprocessing from the original code. Ensure that any data transformations, scaling, encoding, or cleaning procedures are preserved.
Return Statement: Ensure that the function returns the training and test datasets as pandas DataFrames in the format:
`return X_train, y_train, X_test, y_test`
Execution: Include a call to `extract_datasets()` at the end of the code so that the function is executed when the script runs.

Here is an example of how this should be done:
```python
source_code_information = 
{{
    "dataset_name": "diabetes",
    "dataset_library": "sklearn",
    "model_type": "SVM",
    "model_source": "sklearn",
    "test_size": 0.3,
    "objective": "classification",
    "number_of_classes": 9,
    "cross_validation": None,
    "feature_selection": None,
    "number_of_features": None,
    "number_of_samples": None,
    "feature_to_sample_ratio": None,
    "linearity_score": None,
    "hyperparameters": "{{'kernel': 'rbf', 'C': 6, 'gamma': 'auto', 'coef0': 1}}",
    "evaluation_metrics": "accuracy",
    "dataset_scaling": "StandardScaler",
    "imputation": "mean imputation",
    "encoding": "LabelEncoder",
    "extract_datasets_function": "from sklearn import datasets\\nfrom sklearn.model_selection import train_test_split\\nfrom sklearn.svm import SVC\\nfrom sklearn.preprocessing import StandardScaler, LabelEncoder\\nfrom sklearn.metrics import accuracy_score, classification_report\\nimport pandas as pd\\n\\ndef extract_datasets():\\n    # Step 1: Load the dataset\\n    data = datasets.load_diabetes()\\n    X = pd.DataFrame(data.data, columns=data.feature_names)\\n    y = pd.Series(data.target, name='target')\\n\\n    # Limit dataset to top 10,000 samples if larger\\n    max_samples = 50000\\n    if len(X) > max_samples:\\n        X, _, y, _ = train_test_split(X, y, train_size=max_samples, stratify=y, random_state=42)\\n\\n    # Step 2: Preprocess the dataset\\n    # Fill missing values only for numerical columns\\n    numeric_cols = X.select_dtypes(include=['number']).columns\\n    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())\\n\\n    # Encode categorical columns\\n    for col in X.select_dtypes(include='object').columns:\\n        X[col] = LabelEncoder().fit_transform(X[col])\\n\\n    # Step 3: Split the dataset into training and test sets\\n    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)\\n\\n    # Step 4: Scale the features\\n    scaler = StandardScaler()\\n    X_train = scaler.fit_transform(X_train)\\n    X_test = scaler.transform(X_test)\\n\\n    return X_train, y_train, X_test, y_test"
}}


Return your findings as a dictionary within triple backticks for Python, formatted as follows, ensuring JSON compatibility by using \\ for escape characters and {{ }} to indicate a literal {{:

{{
    "dataset_name": "<dataset name>",
    "dataset_library": "<dataset library>",
    "model_type": "<model name>",
    "model_source": "<model source library>",
    "test_size": "<test set size>",
    "objective": "<objective type: classification/regression>",
    "number_of_classes": "<number of unique target classes>",
    "cross_validation": "<cross-validation strategy>",
    "feature_selection": "<feature selection techniques used>",
    "number_of_features": "<number of features>",
    "number_of_samples": "<number of samples>",
    "feature_to_sample_ratio": "<feature-to-sample ratio>",
    "linearity_score": "<linearity score>",
    "hyperparameters": "<python dict: hyperparameters>",
    "evaluation_metrics": "<evaluation metrics>",
    "dataset_scaling": "<scaling techniques>",
    "imputation": "<imputation strategy>",
    "encoding": "<encoding techniques>",
    "extract_datasets_function": "<function code>"
}}
"""