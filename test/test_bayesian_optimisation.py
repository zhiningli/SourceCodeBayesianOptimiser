import unittest
import numpy as np
from src.surrogate.kernels.RBF import RBF
from src.acquisition.EI import EI
from src.surrogate.GP import GP
from src.optimiser.optimiser import Optimiser

def black_box_function(x):
    return np.sin(5 * x) * (1 - np.tanh(x ** 2))

class TestGPOptimiser(unittest.TestCase):
    
    def test_optimisation_with_gp(self):

        X_init = np.random.uniform(-2, 2, (5, 1))
        y_init = black_box_function(X_init)

        kernel = RBF(length_scale=0.5)
        model = GP(kernel=kernel)
        acquisition = EI(xi=0.02)

        bounds = [(-2.0, 2.0)]

        optimiser = Optimiser(acquisition=acquisition, model=model, n_iter=5, objective_func=black_box_function)

        result = optimiser.optimise(X_init, y_init, bounds)

        best_point = result['best_point']
        best_value = result['best_value']

        self.assertTrue(-2.0 <= best_point <= 2.0, "Best point is out of bounds")

        self.assertGreater(best_value, np.max(y_init), "Optimiser did not find a better value than the initial training data")
    
    def test_acquisition_computation(self):
        X_init = np.random.uniform(-2, 2, (5, 1))
        y_init = black_box_function(X_init)

        kernel = RBF(length_scale=0.5)
        gp = GP(kernel=kernel)
        gp.fit(X_init, y_init)

        acquisition = EI(xi=0.01)

        X_candidates = np.linspace(-2, 2, 100).reshape(-1, 1)
        ei_values = acquisition.compute(X_candidates, gp)

        self.assertTrue(np.all(ei_values >= 0), "EI values should be non-negative")
        self.assertGreater(np.max(ei_values), 0, "Expected Improvement values should have a peak")

        next_sample = X_candidates[np.argmax(ei_values)]
        self.assertTrue(-2 <= next_sample <= 2, "Next sample point is out of bounds")

if __name__ == '__main__':
    unittest.main()