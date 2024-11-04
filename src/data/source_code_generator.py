from src.data.data_models import SVMHyperParameterSpace, BOHyperparameterSpace

class SourceCode:

    def __init__(self, source_code_hyperparameters=None, optimalBOHyperParameters=None):
        self.optimalBOHyperParameters = optimalBOHyperParameters
        self.source_code_hyperparameters = source_code_hyperparameters


##################################################################################################################
#  Structured source code that generated from various libraries' dataset, using SVC model from SK-learn          #
##################################################################################################################
class SVMSourceCode(SourceCode):

    def __init__(self):
        super().__init__(source_code_hyperparameters={
            "kernel": "",
            "C": "",
            "gamma": "",
            "coef0": ""
        })

        self.dataset_name = None
        self.dataset_id = None
        self.library = None
        self.name = None
        self.source_code_type = "SVM"

        self.SVMHyperparameters_searchSpace = SVMHyperParameterSpace
        self.BOHyperparameters_searchSpace = self._generateBOHyperparameterSearchSpace()
        
        self.optimalSVMHyperparameter = None
        self.optimalBOHyperParameters = None

    
    @classmethod
    def builder(cls):
        return SVMSourceCodeBuilder()

    @property
    def get_source_code(self):
        dataImportSourceCode = {
    "sklearn": {
        "importText": """
from sklearn import datasets
import pandas as pd
""",
        "loadDataText": f"""
    data = datasets.load_{self.dataset_name}()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    target_names = [str(name) for name in data.target_names]
"""
    },
    "openml": {
        "importText": """
import openml
import pandas as pd
""",
        "loadDataText": f"""
    dataset = openml.datasets.get_dataset(dataset_id={self.dataset_id})
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute, dataset_format="dataframe")
    y = pd.Series(y, name='target')
    target_names = dataset.retrieve_class_labels() if dataset.retrieve_class_labels() else None
"""
    }
}

        # Select the appropriate data import code based on the library
        data_loading_code = dataImportSourceCode.get(self.library, "")

        # Initialize SVM model with hyperparameters
        # model_initialization = f"model = SVC(kernel = kernel, C = C"
        # if self.source_code_hyperparameters["kernel"] in ["poly", "rbf", "sigmoid"]:
        #     model_initialization += f", gamma = gamma"
        # if self.source_code_hyperparameters["kernel"] in ["poly", "sigmoid"]:
        #     model_initialization += f", coef0 = coef0"

    # Generate complete source code
        return f"""
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
{data_loading_code["importText"]}

def run_svm_classification(kernel, C, coef0, gamma):
    # Step 1: Load the dataset
    {data_loading_code["loadDataText"]}

    # Limit dataset to top 10,000 samples if larger
    max_samples = 10000
    if len(X) > max_samples:
        X, _, y, _ = train_test_split(X, y, train_size=max_samples, stratify=y, random_state=42)

    # Step 2: Preprocess the dataset
    # Fill missing values only for numerical columns
    numeric_cols = X.select_dtypes(include=['number']).columns
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

    # Encode categorical columns
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])
        
    # Step 3: Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 4: Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 5: Initialize the SVM model with hyperparameters
    model = SVC(kernel=kernel, C = C, gamma= gamma, coef0 = coef0)

    # Step 6: Train the model
    model.fit(X_train, y_train)

    # Step 7: Make predictions on the test set
    y_pred = model.predict(X_test)

    # Step 8: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    print("Model Accuracy:", accuracy)
    print("\\nClassification Report:\\n", report)

    return accuracy
    """

    
    def get_optimal_BO_hyperParameters(self):
        if not self.optimalBOHyperParameters:
            return "Optimal BO HyperParameters not implemented yet"
        return self.optimalBOHyperParameters
    
    def set_optimal_BO_hyperParameters(self, BOHyperparameters):
        pass
    
    def set_SVMhyperparameters(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.SVMHyperparameters_searchSpace:
                if self.SVMHyperparameters_searchSpace[key]["type"] == "categorical" and value in self.SVMHyperparameters_searchSpace[key]["options"]:
                    self.source_code_hyperparameters[key] = value
                elif self.SVMHyperparameters_searchSpace[key]["type"] == "continuous" and self.SVMHyperparameters_searchSpace[key]["range"][0] <= float(value) <= self.SVMHyperparameters_searchSpace[key]["range"][1]:
                    self.source_code_hyperparameters[key] = value
                else:
                    raise ValueError("Check the datatype of the input parameters")
    
    def _generateBOHyperparameterSearchSpace(self):
        bOHyperparameterSpace = BOHyperparameterSpace

        for kernel in bOHyperparameterSpace["GPHyperParameter"]["kernel"]["options"]:
            for dim in ["kernel", "gamma","C", "coef0"]:
                bOHyperparameterSpace["GPHyperParameter"]["kernel"][kernel]["length_scale"][dim] = self.SVMHyperparameters_searchSpace[dim]         
        return bOHyperparameterSpace

    def generate_name(self):
        return f"""SVM_source_code_on_{self.library}_{self.dataset_name if self.dataset_name else ""}{self.dataset_id if self.dataset_id else ""}_dataset"""


class SVMSourceCodeBuilder:
    """Builder class for constructing an SVMSourceCode object step-by-step."""

    def __init__(self):
        self._svm_source_code = SVMSourceCode()

    def buildKernel(self, kernel: str):
        self._svm_source_code.set_SVMhyperparameters(kernel=kernel)
        return self

    def buildC(self, C: float):
        self._svm_source_code.set_SVMhyperparameters(C=C)
        return self

    def buildGamma(self, gamma: str):
        self._svm_source_code.set_SVMhyperparameters(gamma=gamma)
        return self
    
    def buildCoef0(self, coef0: float):
        self._svm_source_code.set_SVMhyperparameters(coef0=coef0)
        return self

    def buildDataSet(self, library: str, dataset_id: int = None, dataset_name = None):
        """Set the library and dataset ID if applicable."""
        self._svm_source_code.library = library
        self._svm_source_code.dataset_id = dataset_id
        self._svm_source_code.dataset_name = dataset_name

        self._svm_source_code.name = self._svm_source_code.generate_name()
        return self

    def build(self) -> SVMSourceCode:
        """Return the fully constructed SVMSourceCode object."""
        return self._svm_source_code

