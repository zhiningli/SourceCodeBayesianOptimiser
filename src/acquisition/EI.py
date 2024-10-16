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
        
        # Predict mean and std from the model for each candidate point
        mean, std = model.predict(X)
        
        # Ensure mean and std are 1D arrays
        mean = mean.reshape(-1)
        std = std.reshape(-1)  # std should have the same shape as mean

        # Compute the best observed value
        mean_opt = np.max(model.y_train)
        
        # Calculate Z value (should be 1D)
        Z = (mean - mean_opt - self.xi) / (std + 1e-9)
        
        # Use the custom norm to calculate PDF and CDF
        norm = Norm()
        ei = (mean - mean_opt - self.xi) * norm.cdf(Z) + std * norm.pdf(Z)
        
        # Print shapes for debugging
        print(f"Shape of mean: {mean.shape}")
        print(f"Shape of std: {std.shape}")
        print(f"Shape of Z: {Z.shape}")      
        print(f"Shape of ei_values: {ei.shape}")

        # Ensure the result is a 1D array for indexing purposes
        ei = ei.flatten()
        return ei

