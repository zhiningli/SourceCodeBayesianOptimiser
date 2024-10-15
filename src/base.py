from enum import Enum

class ModelName(Enum):
    GP = "Gaussian Processes" 


class Model:
    
    def __init__(self):
        pass

    def fit(self, X, y):
        raise NotImplementedError("Must be implemented by subclass")
    
    def predict(self, X):
        raise NotImplementedError("Must be implemented by subclass")
    


class Kernel:
       
    def __call__(self, X, y):
        raise NotImplementedError("Must be implemented by subclass")
    
    
class Acquisition:

    def __init__(self):
        pass

    def compute(self, X, y):
        raise NotImplementedError("Must be implemented by subclass")