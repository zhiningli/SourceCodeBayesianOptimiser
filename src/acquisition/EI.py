from src.base import Acquisition
from src.surrogate.GP import GP
import numpy as np
from src.utils.norm import Norm

class EI(Acquisition):

    def __init__(self, xi=0.01):
        self.xi = xi

    def compute(self, X, model: GP):
        norm = Norm()
        mean, std = model.predict(X)
        std = std.reshape(-1, 1)
        mean_opt = np.max(model.y_train)
        Z = (mean - mean_opt - self.xi) / (std + 1e-9)
        ei = (mean - mean_opt - self.xi) * norm.cdf(Z) + std * norm.pdf(Z)
        return ei.flatten