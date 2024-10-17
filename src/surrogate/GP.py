from src.base import Kernel, Model
import numpy as np

class GP(Model):
    
    def __init__(self, kernel: Kernel, noise=1e-2):
        self.kernel = kernel
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.y_mean = None
        self.y_std = None
        self.L = None
    
    def __str__(self):
        return f"Gaussian Process surrogate utilizing a {self.kernel}"
    
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_mean = y.mean()
        self.y_std = y.std()
        self.y_train = (y - self.y_mean) / self.y_std
        
        K = self.kernel(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
        
        for attempt in range(5):
            try:
                self.L = np.linalg.cholesky(K)
                break
            except np.linalg.LinAlgError:
                self.noise *= 10  # Increase noise for stability
                K = self.kernel(self.X_train, self.X_train) + self.noise * np.eye(len(self.X_train))
        else:
            print("Cholesky decomposition failed. Using inverse calculation as fallback.")
            self.K_inv = np.linalg.inv(K)
    
    def predict(self, X):
 
        K_s = self.kernel(self.X_train, X)
        K_ss = self.kernel(X, X) + self.noise * np.eye(len(X))
        
        # Solve for alpha using the Cholesky decomposition
        alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y_train))
        mean = K_s.T @ alpha
        
        # No need to rescale 'mean' here; keep it in the standardized space
        # mean = mean * self.y_std + self.y_mean  # Remove this line
        
        v = np.linalg.solve(self.L, K_s)
        cov = K_ss - v.T @ v
        
        mean = mean.ravel()
        std = np.sqrt(np.maximum(np.diag(cov), 1e-2))  # Avoid overly small variances
        return mean, std


