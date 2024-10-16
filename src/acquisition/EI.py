import numpy as np
from src.base import Acquisition
from src.surrogate.GP import GP
from src.utils.norm import Norm

class EI(Acquisition):

    def __init__(self, xi=0.01):
        self.xi = xi

    def compute(self, X, model: GP):
        """
        Compute the Expected Improvement (EI) at points X based on the given model.
        
        Parameters:
        X (array-like): Candidate points, shape (num_samples, num_features).
        model (GP): The Gaussian Process model used as a surrogate.

        Returns:
        np.ndarray: Expected improvement values for each candidate, shape (num_samples,).
        """
        X = np.atleast_2d(X)  # Ensure X is at least 2D
        mean, std = model.predict(X)
        mean = mean.reshape(-1)
        std = std.reshape(-1) 
        mean_opt = np.max(model.y_train)
        Z = (mean - mean_opt - self.xi) / (std + 1e-9)
        norm = Norm()
        ei = (mean - mean_opt - self.xi) * norm.cdf(Z) + std * norm.pdf(Z)
        ei = ei.flatten()
        return ei

