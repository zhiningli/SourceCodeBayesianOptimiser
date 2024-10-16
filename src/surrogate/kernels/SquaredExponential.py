import numpy as np
from src.base import Kernel

class SE(Kernel):

    def __init__(self, length_scale = 1.0):
        self.length_scale = length_scale

    def __call__(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        dists = np.sum((X1[:, np.newaxis] - X2[np.newaxis, :])**2, axis = 2)
        return np.exp(-0.5 * dists / self.length_scale **2)
    
