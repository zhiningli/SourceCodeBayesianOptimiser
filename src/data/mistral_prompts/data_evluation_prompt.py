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
Data Importing and Preprocessing: Include all steps related to data loading, importing, and preprocessing from the original code. Ensure that any data transformations, scaling, encoding (one hot or standard), or cleaning procedures are preserved as the source code itself is guaranteed to be runnable.
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


extract_dataset_from_source_code_prompt = """
Your task is to analyze the provided source code and create a new function that extracts and returns the training and test datasets.

<code>
{source_code}
</code>

`return X_train, y_train, X_test, y_test`
Execution: Include a call to `extract_datasets()` at the end of the code so that the function is executed when the script runs.
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

Here is an example of how this should be done. Pay attention to ensuring JSON compatibility by using \\ for escape characters and using double quotes "" for property names.

```python
{{
    "dataset_name": "iris",
    "dataset_library": "sklearn",
    "model_type": "SVM",
    "model_source": "sklearn",
    "objective": "classification",
    "test_size": 0.3,
    "cross_validation": "train test split",
    "feature_selection": "principle feature analysis",
    "hyperparameters": {{
        "kernel": "poly",
        "C": 4.229101316230308,
        "gamma": "scale",
        "coef0": 0.47320558504355414,
        "random_state": 42
    }},
    "dataset_scaling": "StandardScaler",
    "evaluation_metrics": "accuracy",
    "imputation": "mean imputation",
    "encoding": "LabelEncoder"
}}
Return your findings as a dictionary within triple backticks for Python, formatted as follows:
{{
    "dataset_name": "<dataset name>",
    "dataset_library": "<dataset library>",
    "model_type": "<model name>",
    "model_source": "<library if it is an import>",
    "objective": "<classification or regression or else>",
    "test_size": "<proportion used for testing>",
    "cross_validation": "<if any>",
    "feature_selection": "<if any>",
    "hyperparameters": {{
        "key1": "value1",
        "key2": "value2"
    }},
    "dataset_scaling": "<scaling techniques>",
    "evaluation_metrics": "<evaluation metrics>",
    "imputation": "<imputation methods>",
    "encoding": "<Any encoder used>"
}}
"""


extract_information_from_script_prompts = """
Your task is to analyze and extract relevant sections from the provided source code.
<code>
{source_code}
</code> 
Please help me extract the following information:

1. "The section that contains the model architecture"
2. "The section that contains the dataset information"

Here is an example of how this should be done. Pay attention to the <> used as it will be used for parsing later "

```python
<model>
class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        
        # Reshape the input to (batch_size, 1, input_size) for Conv1d
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * (input_size // 2), 128)  # Adjust for the reduced dimension after pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Add regularization with dropout

    def forward(self, x):
        # Add a channel dimension for Conv1d (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
</model>

<dataset>
X, y = make_classification(
    n_samples=4000,          # Number of samples
    n_features=25,           # Total number of features
    n_informative=10,        # Number of informative features
    n_redundant=2,           # Number of redundant features
    n_classes=3,             # Number of classes
    n_clusters_per_class=2,  # Increase clusters per class for complexity
    class_sep=0.5,           # Reduce class separation for overlap
    flip_y=0.1,              # Introduce noise in labels
    random_state=42          # Random state for reproducibility
)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Create DataLoaders
train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # Reusing the same dataset for simplicity
</dataset>

Notes:
1. Ensure only the relevant sections are extracted and formatted exactly as shown above.
2. If a section is missing, indicate it like this

<model>
# No model architecture found
</model>

3. Sometimes models are consisted of multiple classes, extract them together like follows:
<model>
class ResidualBlock(nn.Module):
    def __init__(self, input_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_size, input_size)

    def forward(self, x):
        residual = x
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x += residual  # Skip connection
        return x

class Model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.fc2(x)
        return x
</model>
4. Ensure the extracted sections are properly formatted and adhere to Python syntax.
5. Do not include any additional explanations or comments outside the required sections. 

Return your findings as a python script like above:
<model>
(the model that used in the code str)
</model>

<dataset>
(the dataset that used in the code str)
</dataset>
"""