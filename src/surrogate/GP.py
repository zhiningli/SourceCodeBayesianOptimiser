from src.base import Kernel, Model
import numpy as np

class GP(Model):
    
    def __init__(self, kernel: Kernel, noise=1e-2):
        self.kernel = kernel
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
    
    def __str__(self):
        return f"Gaussian Process surrogate utilising a {self.kernel}"

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        K = self.kernel(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
        self.K_inv = np.linalg.inv(K)

    
    def predict(self, X):
        K_s = self.kernel(self.X_train, X)
        K_ss = self.kernel(X, X)
        mean = K_s.T @ self.K_inv @ self.y_train
        cov = K_ss - K_s.T @ self.K_inv @ K_s
        return mean, np.diag(cov)
    

