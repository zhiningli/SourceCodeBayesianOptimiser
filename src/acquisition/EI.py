import numpy as np
from src.base import Acquisition
from src.surrogate.GP import GP
from scipy.stats import norm

class EI(Acquisition):
    def __init__(self, xi):
        self.xi = xi

    def compute(self, X, model: GP):
        X = np.atleast_2d(X)
        mean, std = model.predict(X)
        
        # Ensure mean_opt is in the same scale as 'mean'
        mean_opt = np.min(model.y_train)  # No need for rescaling as mean is standardized in predict.
        
        # Clip standard deviation to avoid numerical issues
        std = np.clip(std, 1e-9, None)
        
        # Calculate Z for the Expected Improvement formula
        Z = (mean - mean_opt - self.xi) / std
        cdf_values = norm.cdf(Z)
        pdf_values = norm.pdf(Z)
        
        # Compute Expected Improvement
        ei = (mean - mean_opt - self.xi) * cdf_values + std * pdf_values
        
        # Avoid negative EI values due to numerical errors
        ei = np.maximum(ei, 0)
        return ei


