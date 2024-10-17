import numpy as np
from src.base import Kernel

class RBF(Kernel):
    def __init__(self, length_scales):
        """
        Anisotropic RBF kernel with different length scales for each dimension.
        
        Parameters:
        length_scales (array-like): A list or array of length scales, one for each dimension.
        """
        self.length_scales = np.array(length_scales)

    def __call__(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        # Compute the squared Euclidean distance, scaled by the length scales.
        scaled_dists = np.sum(
            ((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2) / self.length_scales**2,
            axis=2
        )

        # Compute the RBF kernel.
        return np.exp(-0.5 * scaled_dists)

    def __str__(self):
        return f"Anisotropic RBF Kernel with length scales {self.length_scales}"
