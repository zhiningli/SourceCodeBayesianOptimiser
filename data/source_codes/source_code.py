from hyperparameterSpace import SVMHyperParameterSpace
from sklearn.svm import SVC
from typing import List, Dict, Any
from sklearn import datasets
from sklearn.model_selection import train_test_split

class SourceCode:

    def __init__(self, description, optimalBayesianHyperparameters):
        pass

class SVMSourceCode(SourceCode):

    def __init__(self, dataset_name: str = ""):
        super(SourceCode).__init__()
        self.optimalBOHyperParameters = None
        self.SVMHyperparameters_searchSpace = SVMHyperParameterSpace
        self.hyperparameters = {
            "kernel": "",
            "C": "",
            "gamma": "",
            "coef0": ""
        }
        self.dataset_name = self.set_dataset_name(dataset_name)
        self.dataset_loaded = False
        self.X_train, self.X_Test, self.y_train, self.y_test = None, None, None, None
    
    @classmethod
    def builder(cls):
        return SVMSourceCodeBuilder()

    @property
    def get_source_code(self):
        model_initialization = f"model = SVC(kernel='{self.hyperparameters['kernel']}', C={self.hyperparameters['C']}"

        if self.hyperparameters["kernel"] in ["poly", "rbf", "sigmoid"]:
            model_initialization += f", gamma='{self.hyperparameters['gamma']}'"
        if self.hyperparameters["kernel"] in ["poly", "sigmoid"]:
            model_initialization += f", coef0={self.hyperparameters['coef0']}"

        model_initialization += ", random_state=42)"

        return f"""
def run_svm_classification():
    # Step 1: Load the dataset
    data = load_{self.dataset_name}()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names

    # Step 2: Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Step 3: Initialize the SVM model with hyperparameters
    {model_initialization}

    # Step 4: Train the model
    model.fit(X_train, y_train)

    # Step 5: Make predictions on the test set
    y_pred = model.predict(X_test)

    # Step 6: Evaluate the model
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
                    self.hyperparameters[key] = value
                elif self.SVMHyperparameters_searchSpace[key]["type"] == "continuous" and self.SVMHyperparameters_searchSpace[key]["range"][0] <= float(value) <= self.SVMHyperparameters_searchSpace[key]["range"][1]:
                    self.hyperparameters[key] = value
                else:
                    raise ValueError("Check the datatype of the input parameters")
                
    
    def set_dataset_name(self, dataset_name: str):
        if dataset_name == "":
            return
        dataset_names = set(["iris", "digits", "wine", "diabetes", "breast_cancer", "svmlight_files", "files", "linnerud", "sample_images"])
        if dataset_name in dataset_names:
            self.dataset = dataset_name
        else:
            raise ValueError(f"Dataset {dataset_name} is not recognised. Currently only scikit learn datasets are supported")
        
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
        elif self.dataset_name == "svmlight_files":
            data = datasets.load_svmlight_files() 
        elif self.dataset_name == "files":
            data = datasets.load_files()
        elif self.dataset_name == "linnerud":
            data = datasets.load_linnerud()
        elif self.dataset_name == "sample_images":
            data = datasets.load_sample_images() 
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' not recognized.")
        
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.dataset_loaded = True

    def create_model(self) -> SVC:
        self._load_dataset()
        return SVC(**self.hyperparameters)



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

    def buildDataSet(self, dataset_name: str):
        """Set only the dataset name for lazy loading later."""
        self._svm_source_code.set_dataset_name(dataset_name)
        return self

    def build(self) -> SVMSourceCode:
        """Return the fully constructed SVMSourceCode object."""
        return self._svm_source_code

