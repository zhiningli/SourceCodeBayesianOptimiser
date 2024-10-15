import unittest
import numpy as np
from src.kernels.RBF import RBF
from src.acquisition.EI import EI
from src.surrogate.GP import GP

def black_box_function(x):
    return np.sin(5 * x) * (1 - np.tanh(x ** 2))


class TestBayesianOptimization(unittest.TestCase):
    def test_optimization(self):
        X_init = np.random.uniform(-2, 2, (5, 1))
        y_init = black_box_function(X_init)

        kernel = RBF(length_scale=0.5)
        gp = GP(kernel=kernel)
        gp.fit(X_init, y_init)

        acquisition = EI(xi=0.01)

        X_candidates = np.linspace(-2, 2, 100).reshape(-1, 1)
        ei_values = acquisition.compute(X_candidates, gp)

        next_sample = X_candidates[np.argmax(ei_values)]

        self.assertTrue(-2 <= next_sample <= 2, "Next sample point is out of bounds")
        self.assertGreater(np.max(ei_values()), 0, "Expected Improvement values should be positive")


if __name__ == '__main__':
    unittest.main()
if __name__ == '__main__':
    unittest.main()