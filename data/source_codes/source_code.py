from data.source_codes.hyperparameterSpace import SVMHyperParameterSpace
from sklearn.svm import SVC
from typing import List, Dict, Any
from sklearn import datasets
from sklearn.model_selection import train_test_split
import openml

class SourceCode:

    def __init__(self, source_code_hyperparameters=None, optimalBOHyperParameters=None):
        self.optimalBOHyperParameters = optimalBOHyperParameters
        self.source_code_hyperparameters = source_code_hyperparameters


##################################################################################################################
#  Structured source code generator that generated from SK-learn library dataset, using SVC model from SK-learn  #
##################################################################################################################
class SVMSourceCode(SourceCode):

    def __init__(self, dataset_name: str = "", library: str = "sklearn", dataset_id: int = None):
        super().__init__(source_code_hyperparameters={
            "kernel": "",
            "C": "",
            "gamma": "",
            "coef0": ""
        })
        self.optimalBOHyperParameters = None
        self.SVMHyperparameters_searchSpace = SVMHyperParameterSpace
        self.dataset_name = dataset_name
        self.dataset_id = dataset_id
        self.library = library
        self.dataset_loaded = False
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
    
    @classmethod
    def builder(cls):
        return SVMSourceCodeBuilder()

    @property
    def get_source_code(self):
        # Define data import code snippets for different libraries
        dataImportSourceCode = {
            "sklearn": {
                "importText": f"""
from sklearn import datasets
""", 
                "loadDataText": 
                f"""
    data = datasets.load_{self.dataset_name}()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names
"""},
            "openml": {
                "importText": f"""
import openml
""", 
                "loadDataText": 
                f"""
    dataset = openml.datasets.get_dataset({self.dataset_id})
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    feature_names = dataset.feature_names
    target_names = dataset.target_names
"""}
        }
        
        # Select the appropriate data import code based on the library
        data_loading_code = dataImportSourceCode.get(self.library, "")

        # Initialize SVM model with hyperparameters
        model_initialization = f"model = SVC(kernel='{self.source_code_hyperparameters['kernel']}', C={self.source_code_hyperparameters['C']}"
        if self.source_code_hyperparameters["kernel"] in ["poly", "rbf", "sigmoid"]:
            model_initialization += f", gamma='{self.source_code_hyperparameters['gamma']}'"
        if self.source_code_hyperparameters["kernel"] in ["poly", "sigmoid"]:
            model_initialization += f", coef0={self.source_code_hyperparameters['coef0']}"
        model_initialization += ", random_state=42)"

        # Generate complete source code
        return f"""
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
{data_loading_code["importText"]}

def run_svm_classification():
    # Step 1: Load the dataset
    {data_loading_code["loadDataText"]}

    # Step 2: Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 3: Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Step 4: Initialize the SVM model with hyperparameters
    {model_initialization}

    # Step 5: Train the model
    model.fit(X_train, y_train)

    # Step 6: Make predictions on the test set
    y_pred = model.predict(X_test)

    # Step 7: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    print("Model Accuracy:", accuracy)
    print("\\nClassification Report:\\n", report)
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

    def _load_dataset(self):
        """Lazy loads the dataset based on the dataset name."""
        if self.dataset_loaded:
            return
        if self.dataset_name == "iris":
            data = datasets.load_iris()
        elif self.dataset_name == "digits":
            data = datasets.load_digits()
        elif self.dataset_name == "wine":
            data = datasets.load_wine()
        elif self.dataset_name == "diabetes":
            data = datasets.load_diabetes()
        elif self.dataset_name == "breast_cancer":
            data = datasets.load_breast_cancer()
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' not recognized.")
        
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.dataset_loaded = True

    def create_model(self) -> SVC:
        self._load_dataset()
        return SVC(**self.source_code_hyperparameters)


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
        return self

    def build(self) -> SVMSourceCode:
        """Return the fully constructed SVMSourceCode object."""
        return self._svm_source_code

