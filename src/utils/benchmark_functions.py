import numpy as np
from typing import List, AnyStr
import inspect

class BenchmarkFunctions:
    def __init__(self, n_dimension: int, 
                 search_space_ranges: np.array, 
                 global_minimum: float, 
                 global_minimumX: np.array, 
                 noises: float = 0.0, 
                 irrelevant_dims: int = 0, 
                 irrelevant_dims_search_space: np.array = np.array([]),
                 description=""):
        """
        n_dimension: Number of relevant dimensions.
        search_space_ranges: A numpy array specifying the search space for each input variables np.array([[lowerX1, upperX1], [lowerX2, upperX2], ...]) ).
        global_minimum: The known global minimum value.
        global_minimumX: The known global minimum location for relevant dimensions.
        description: Optional description of the benchmark function.
        """
        self.n_dimension = n_dimension
        self._search_space_ranges = search_space_ranges
        self._global_minimum = global_minimum
        self._global_minimumX = global_minimumX
        self.description = description
        self.noise_std = noises
        self.irrelevant_dims = irrelevant_dims
        self.irrelevant_search_spaces = irrelevant_dims_search_space


    @property
    def search_space(self) -> np.array:
        """
        Return the search space for the benchmark function as a numpy array, including irrelevant dimensions.
        """
        search_space = self._search_space_ranges.copy()
        if self.irrelevant_dims > 0:
            search_space = np.concatenate((search_space, self.irrelevant_search_spaces), axis=0)
        return search_space
    
    @property
    def global_minimum(self) -> float:
        """
        Return the known global minimum value.
        """
        return self._global_minimum
    
    @property
    def global_minimumX(self) -> np.array:
        """
        Return the known global minimum location, extended with irrelevant dimensions (if any).
        For irrelevant dimensions, a random value within the irrelevant search space will be used.
        """
        return np.array(self._global_minimumX) 
    
    @property
    def describe(self) -> AnyStr:
        """
        Return the description of the benchmark function.
        """
        return self.description
    
    def _source_code(self):
        """The actual source code that should be read by our source code reader"""
        pass

    def set_noise(self, noise_std) -> float:
        """
        Set the standard deviation of the Gaussian noise to be added.
        """
        self.noise_std = noise_std
    
    def add_irrelevant_dimensions(self, n_irrelevant: int, irrelevant_search_spaces: np.array):
        """
        Add irrelevant (useless) dimensions to the search space, specifying the range for each irrelevant dimension.
        irrelevant_search_spaces: A list of tuples specifying the search range for each irrelevant dimension.
        These dimensions do not affect the objective function value.
        """
        if len(irrelevant_search_spaces) != n_irrelevant:
            raise ValueError("Number of irrelevant search spaces must match the number of irrelevant dimensions.")
        self.irrelevant_dims = n_irrelevant
        self.irrelevant_search_spaces = irrelevant_search_spaces
    
    def print_dimension_info(self) -> AnyStr:
        """
        Print the dimension of the search space, including irrelevant dimensions.
        """
        total_dims = self.n_dimension + self.irrelevant_dims
        print(f"The search space has {self.n_dimension} relevant dimensions and {self.irrelevant_dims} irrelevant dimensions, totaling {total_dims} dimensions.")
    
    def evaluate(self, X):
        """
        To be inherited by benchmark functions, this is the source code that is going to be passed to the optimser as priors
        """

    def evalute_with_noise(self, X):
        return self.evaluate(X) + np.random.normal(0, self.noise_std)
    

    def get_source_code(self) -> AnyStr:
        """
        Returns the source code of the evaluate function as a string.
        """
        return inspect.getsource(self._source_code)



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
        A = 3
        return A * X.shape[-1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=-1)

    def evaluate(self, X, A=3):
        """
        Evaluates the Rastrigin function for a given input X.

        X: Input can be a 1D, 2D, 3D or higher-dimensional input for the Rastrigin function.
        A: The scale factor (typically set to 10).
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Input X must be a numpy array")


        total_dims = self.n_dimension + self.irrelevant_dims
        if X.shape[-1] != total_dims:
            raise ValueError(f"Input dimension mismatch: expected {total_dims} dimensions but got {X.shape[-1]} dimensions.")
    
        if X.ndim == 1:
            return A * len(X) + np.sum(X**2 - A * np.cos(2 * np.pi * X))
        elif X.ndim == 2:
            return A * X.shape[1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)
        elif X.ndim == 3:
            X, Y = X
            return A * 2 + (X**2 - A * np.cos(2 * np.pi * X)) + (Y**2 - A * np.cos(2 * np.pi * Y))
        else:
            return A * X.shape[-1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=-1)
        


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
        x, y = X[..., 0], X[..., 1]
        return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2

    def evaluate(self, X):
        """
        Evaluates the Beale function at a given input X.

        X: Can be a single point (1D array), multiple points (2D array), or a meshgrid input (3D for plotting).
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("Input X must be a numpy array")
        total_dims = self.n_dimension + self.irrelevant_dims
        if X.shape[-1] != total_dims:
            raise ValueError(f"Input dimension mismatch: expected {total_dims} dimensions but got {X.shape[-1]} dimensions.")
        
        if X.ndim == 1:
            x, y = X 
            return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2

        elif X.ndim == 2:
            x, y = X[:, 0], X[:, 1]  # Unpack x and y from 2D array
            return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2

        
        elif X.ndim == 3:
            X_mesh, Y_mesh = X
            return (1.5 - X_mesh + X_mesh * Y_mesh)**2 + (2.25 - X_mesh + X_mesh * Y_mesh**2)**2 + (2.625 - X_mesh + X_mesh * Y_mesh**3)**2
        
        else:
            x, y = X[..., 0], X[..., 1] 
            return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2


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
        return np.sum(X**2, axis=-1)
    
    def evaluate(self, X):
        """
        Evaluates the Sphere function at a given point X.
        Supports single point evaluation (1D), multiple points (2D), and meshgrid input (3D).
        Handles higher-dimensional arrays (ndim > 3) as well.
        """

        if not isinstance(X, np.ndarray):
            raise TypeError("Input X must be a numpy array")

        total_dims = self.n_dimension + self.irrelevant_dims
        if X.ndim <= 2 and X.shape[-1] != total_dims:
            raise ValueError(f"Input dimension mismatch: expected {total_dims} dimensions but got {X.shape[-1]} dimensions.")

        if X.ndim == 1:
            return np.sum(X**2)
        
        elif X.ndim == 2:
            return np.sum(X**2, axis=1)

        elif X.ndim == 3:
            return np.sum(X**2, axis=-1)
        else:
            return np.sum(X**2, axis=-1)



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


class Dummy3DFunction(BenchmarkFunctions):

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



    
    