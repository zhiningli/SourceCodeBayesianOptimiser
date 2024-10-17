import unittest
import numpy as np
from src.surrogate.kernels.RBF import RBF
from src.acquisition.PI import PI
from src.surrogate.GP import GP
from src.optimiser.optimiser import Optimiser
from src.utils.benchmark_functions import Sphere, Rastrigin

def black_box_function(x):
    return np.sin(5 * x) * (1 - np.tanh(x ** 2))

import unittest
import numpy as np
from src.surrogate.kernels.RBF import RBF
from src.acquisition.PI import PI
from src.surrogate.GP import GP
from src.optimiser.optimiser import Optimiser
from src.utils.benchmark_functions import Sphere, Rastrigin

class Test_PI_GP_RBF_Optimiser(unittest.TestCase):

    def test_PI_GP_RBF_on_Sphere(self):
        n_dimension = 2
        benchmark = Sphere(n_dimension=n_dimension)

        kernel = RBF(length_scales=[1, 1])
        surrogate = GP(kernel=kernel, noise=1e-2)
        acquisition_func = PI(xi=0.1)
        optimiser = Optimiser(acquisition=acquisition_func, model=surrogate, n_iter=50, objective_func=benchmark.evaluate)

        bounds = benchmark.search_space
        np.random.seed(42)
        # Generate initial points within the search space
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        X_init = np.random.uniform(lower_bounds, upper_bounds, (15, benchmark.n_dimension))
        y_init = np.array([benchmark.evaluate(x) for x in X_init])

        result = optimiser.optimise(X_init, y_init, bounds)

        # Calculate the Euclidean distance between the found point and the known global minimum point
        distance_to_global_minimum = np.linalg.norm(result['best_point'] - benchmark.global_minimumX)
        print(f"Distance to global minimum: {distance_to_global_minimum}")

        # Assert that the distance is below a small threshold (e.g., 0.1)
        self.assertTrue(distance_to_global_minimum < 0.1, "The optimiser did not find a point close enough to the global minimum in the input space.")

    def test_PI_GP_RBF_on_Rastrigin(self):
        n_dimension = 2
        benchmark = Rastrigin(n_dimension=n_dimension)

        kernel = RBF(length_scales=[0.25, 0.25])
        surrogate = GP(kernel=kernel, noise=1e-2)
        acquisition_func = PI(xi=0.1)
        optimiser = Optimiser(acquisition=acquisition_func, model=surrogate, n_iter=300, objective_func=benchmark.evaluate)

        bounds = benchmark.search_space
        np.random.seed(42)   
        # Extract lower and upper bounds for each dimension
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])

        # Generate initial points, ensuring each dimension is sampled within its respective range
        X_init = np.random.uniform(lower_bounds, upper_bounds, (5, benchmark.n_dimension))
        y_init = np.array([benchmark.evaluate(x) for x in X_init])

        result = optimiser.optimise(X_init, y_init, bounds)

        # Calculate the Euclidean distance between the found point and the known global minimum point
        distance_to_global_minimum = np.linalg.norm(result['best_point'] - benchmark.global_minimumX)
        print(f"Distance to global minimum: {distance_to_global_minimum}")

        # Assert that the distance is below a small threshold (e.g., 0.1)
        self.assertTrue(distance_to_global_minimum < 0.1, "The optimiser did not find a point close enough to the global minimum in the input space.")

if __name__ == '__main__':
    unittest.main()


if __name__ == '__main__':
    unittest.main()