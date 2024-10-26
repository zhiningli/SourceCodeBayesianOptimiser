from source_code import SourceCode
import inspect
from sklearn.svm import SVC
from typing import List, Dict, Any
from sklearn import datasets
from sklearn.model_selection import train_test_split

class SVMSourceCode(SourceCode):

    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale', dataset_name: str = ""):
        super(SourceCode).__init__()
        self.optimalBOHyperParameters = None
        self.hyperparameters = {
            'kernel': kernel,
            'C': C,
            'gamma': gamma
        }
        self.dataset_name = dataset_name
        self.dataset_loaded = False
        self.X_train, self.X_Test, self.y_train, self.y_test = None, None, None, None
    
    @classmethod
    def builder(cls):
        return SVMSourceCodeBuilder()
    
    def _source_code(self):
        pass

    def get_source_code(self):
        return inspect(self._source_code)
    
    def set_optimal_BO_hyperParameters(self, BOHyperparameters):
        pass
    
    def set_SVMhyperparameters(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.hyperparameters:
                self.hyperparameters[key] = value
    
    def set_dataset_name(self, dataset_name: str):
        dataset_names = set("iris", "digits", "wine", "diabetes", "breast_cancer", "svmlight_sample", "files", "linnerud", "sample_images")
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
        elif self.dataset_name == "svmlight_sample":
            data = datasets.load_svmlight_files() 
        elif self.dataset_name == "load_files":
            data = datasets.load_files()
        elif self.dataset_name == "load_linnerud":
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

    def buildDataSet(self, dataset_name: str):
        """Set only the dataset name for lazy loading later."""
        self._svm_source_code.set_dataset_name(dataset_name)
        return self

    def build(self) -> SVMSourceCode:
        """Return the fully constructed SVMSourceCode object."""
        return self._svm_source_code