import numpy as np
from src.utils.benchmark_functions.benchmark_functions import BenchmarkFunctions

class Rastrigin(BenchmarkFunctions):

    def __init__(self, n_dimension, noises=0.0, irrelevant_dims=0):
        super().__init__(n_dimension=n_dimension, 
                         search_space_ranges=np.array([(-5.12, 5.12)] * n_dimension), 
                         global_minimum=0,
                         global_minimumX=np.array([0] * n_dimension),
                         noises=noises, 
                         irrelevant_dims=irrelevant_dims,
                         description="""The Rastrigin function is a multi-modal, highly non-convex benchmark function used to test optimization algorithms. 
It has a global minimum at x=0, where the function value is zero. The typical search range for each dimension is between -5.12 and 5.12.""") 
    
    def _source_code(self, X):
        """
        Source code for the Rastrigin function.
        Assumes X is a 1D array representing a single input.
        """
        A = 3
        return A * len(X) + np.sum(X**2 - A * np.cos(2 * np.pi * X))

    def evaluate(self, X, A=3):
        """
        Evaluates the Rastrigin function for a given 1D input X.
        
        Args:
            X (np.ndarray): A 1D array with n_dimension elements.
            A (float): The scale factor (typically set to 10).
        
        Returns:
            float: The computed Rastrigin function value for the input X.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("Input X must be a numpy array")

        total_dims = self.n_dimension + self.irrelevant_dims
        if X.shape[-1] != total_dims:
            raise ValueError(f"Input dimension mismatch: expected {total_dims} dimensions but got {X.shape[-1]} dimensions.")

        return A * len(X) + np.sum(X**2 - A * np.cos(2 * np.pi * X))



class Beale(BenchmarkFunctions):
    def __init__(self, n_dimension=2, noises=0.0, irrelevant_dims=0):
        super().__init__(n_dimension=n_dimension, 
                         search_space_ranges=[(-4.5, 4.5)] * n_dimension, 
                         global_minimum=0,
                         global_minimumX=[3, 0.5],  # Global minimum at (3, 0.5)
                         noises=noises, 
                         irrelevant_dims=irrelevant_dims,
                         description="""The Beale function is a benchmark function with many local minima near the global minimum at (3, 0.5). 
It is used to test optimization algorithms in 2D space. The typical search range is between -4.5 and 4.5 for both x and y.""")
    
    def _source_code(self, X):
        """
        The source code of the Beale function.
        Assumes X is a 1D or 2D array (n_samples, n_features).
        """
        x, y = X[..., 0], X[..., 1]
        return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2

    def evaluate(self, X):
        """
        Evaluates the Beale function at a given input X.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("Input X must be a numpy array")
        
        total_dims = self.n_dimension + self.irrelevant_dims
        if X.shape[-1] != total_dims:
            raise ValueError(f"Input dimension mismatch: expected {total_dims} dimensions but got {X.shape[-1]} dimensions.")
        
        return self._source_code(X)  # Directly return the evaluation


class Sphere(BenchmarkFunctions):
    def __init__(self, n_dimension, noises=0.0, irrelevant_dims=0):
        super().__init__(n_dimension=n_dimension, 
                         search_space_ranges=np.array([(-5.12, 5.12)] * n_dimension), 
                         global_minimum=0,
                         global_minimumX=np.array([0] * n_dimension),
                         noises=noises,
                         irrelevant_dims=irrelevant_dims,
                         description="""The Sphere function is a simple benchmark function for optimization. 
The global minimum is at (0, 0, ..., 0), where f(0) = 0. The typical search space for each dimension 
is [-5.12, 5.12], and the function is convex and unimodal.""")
        
    def _source_code(self, X):
        """
        Source code for the Sphere function. Computes the sum of squares for each input.
        Assumes X is either a 1D or 2D array (n_samples, n_features).
        """
        return np.sum(X**2, axis=-1)
    
    def evaluate(self, X):
        """
        Evaluates the Sphere function at a given input X.
        Supports single point evaluation (1D array) or multiple points (2D array).
        """

        if not isinstance(X, np.ndarray):
            raise TypeError("Input X must be a numpy array")

        total_dims = self.n_dimension + self.irrelevant_dims
        if X.shape[-1] != total_dims:
            raise ValueError(f"Input dimension mismatch: expected {total_dims} dimensions but got {X.shape[-1]} dimensions.")

        return self._source_code(X)




class Function_1(BenchmarkFunctions):

    def __init__(self, n_dimension = 7, noises = 0.0, irrelevant_dims=0):
        super().__init__(
                    n_dimension = n_dimension, 
                    search_space_ranges = np.array([(-5.0, 5.0)]* n_dimension), 
                    global_minimum=0.6, 
                    global_minimumX=np.array([0, 0, 1, 0, 1, 1, 1]), 
                    noises = noises, 
                    irrelevant_dims=irrelevant_dims, 
                    description="""Synthetic binary tree structured function, it has 7 relevant dimensions,
                                first three inputs must be binary, the last 4 inputs can take any values within search space """)

    def _source_code(self, X: np.array):

        X = X[:7]
        r8 = 0.5
        r9 = 0.9

        if X[0] == 0:
            if X[1] == 0:
                val = X[3]**2 + 0.1 + r8
            else:
                val = X[4]**2 + 0.2 + r8
        else:
            if X[2] == 0:
                val = X[5]**2 + 0.3 + r9
            else:
                val = X[6]**2 + 0.4 + r9

        return val    
    
    def evaluate(self, X: np.array):

        if not isinstance(X, np.ndarray):
            raise TypeError("Input X must be a numpy array")

        return self._source_code(X)


class Function_2(BenchmarkFunctions):

    def __init__(self, n_dimension = 3, noises = 0.0, irrelevant_dims=0):
        super().__init__(
                    n_dimension = n_dimension, 
                    search_space_ranges = np.array([(-1.0, 1.0)]*n_dimension), 
                    global_minimum=0, 
                    global_minimumX=np.array([0]*n_dimension), 
                    noises = noises, 
                    irrelevant_dims=irrelevant_dims, 
                    description="""Synthetic binary tree structured function, it has 7 relevant dimensions,
                first three inputs must be binary, the last 4 inputs can take any values within search space """)

    def _source_code(self, X: np.array):
        X = X[:3]
        return (X[0]-0.5)**2 +X[1]**2 + X[2]**2 
    

    def evaluate(self, X: np.array):
        return self._source_code(X)



    
    