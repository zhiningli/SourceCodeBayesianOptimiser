from scipy.stats import norm
import numpy as np
from src.base import Acquisition
from src.surrogate import GP

class PI(Acquisition):
    def __init__(self, xi=0.01):
        self.xi = xi

    def compute(self, X, model: GP):
        X = np.atleast_2d(X)
        mean, std = model.predict(X)
        
        # Ensure mean_opt is in the same scale as 'mean'
        mean_opt = np.min(model.y_train)  # No need for rescaling as mean is standardized in predict.
        
        # Clip standard deviation to avoid numerical issues
        std = np.clip(std, 1e-9, None)
        
        # Calculate Z for the Probability of Improvement formula
        Z = (mean_opt - mean - self.xi) / std
        pi = norm.cdf(Z)
        
        # Avoid negative PI values due to numerical errors
        pi = np.maximum(pi, 0)
        return pi