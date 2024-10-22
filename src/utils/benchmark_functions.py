import numpy as np
import inspect

class BenchmarkFunctions:
    def __init__(self, n_dimension, search_space_ranges, global_minimum, global_minimumX, noises = 0.0, irrelevant_dims = 0,description=""):
        """
        n_dimension: Number of relevant dimensions.
        search_space_ranges: A list of tuples specifying the search range for each dimension (e.g., [(0, 4.5), (-3, 3)]).
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
        self.irrelevant_search_spaces = [] 

    @property
    def search_space(self):
        """
        Return the search space for the benchmark function, including irrelevant dimensions.
        Each dimension can have its own search range.
        """
        search_space = self._search_space_ranges.copy()
        if self.irrelevant_dims > 0:
            search_space += self.irrelevant_search_spaces
        return search_space
    
    @property
    def global_minimum(self):
        """
        Return the known global minimum value.
        """
        return self._global_minimum
    
    @property
    def global_minimumX(self):
        """
        Return the known global minimum location, extended with irrelevant dimensions (if any).
        For irrelevant dimensions, a random value within the irrelevant search space will be used.
        """
        return list(self._global_minimumX) 
    
    @property
    def describe(self):
        """
        Return the description of the benchmark function.
        """
        return self.description
    
    @property
    def _source_code(self):
        """The actual source code that should be read by our source code reader"""
        pass

    def set_noise(self, noise_std):
        """
        Set the standard deviation of the Gaussian noise to be added.
        """
        self.noise_std = noise_std
    
    def add_irrelevant_dimensions(self, n_irrelevant, irrelevant_search_spaces):
        """
        Add irrelevant (useless) dimensions to the search space, specifying the range for each irrelevant dimension.
        irrelevant_search_spaces: A list of tuples specifying the search range for each irrelevant dimension.
        These dimensions do not affect the objective function value.
        """
        if len(irrelevant_search_spaces) != n_irrelevant:
            raise ValueError("Number of irrelevant search spaces must match the number of irrelevant dimensions.")
        self.irrelevant_dims = n_irrelevant
        self.irrelevant_search_spaces = irrelevant_search_spaces
    
    def print_dimension_info(self):
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
    

    def get_source_code(self):
        """
        Returns the source code of the evaluate function as a string.
        """
        return inspect.getsource(self._source_code)


class Rastrigin(BenchmarkFunctions):

    def __init__(self, n_dimension, noises=0.0, irrelevant_dims=0):
        super().__init__(n_dimension=n_dimension, 
                         search_space_ranges=[(-5.12, 5.12)] * n_dimension, 
                         global_minimum=0,
                         global_minimumX=[0] * n_dimension,
                         noises=noises, 
                         irrelevant_dims=irrelevant_dims,
                         description="""The Rastrigin function is a multi-modal, highly non-convex benchmark function used to test optimization algorithms. 
It has a global minimum at x=0, where the function value is zero. The typical search range for each dimension is between -5.12 and 5.12.""") 
    
    @property
    def _get_source_code(self, X):
        A = 3
        return A * X.shape[-1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=-1)

    def evaluate(self, X, A=3):
        """
        Evaluates the Rastrigin function for a given input X.

        X: Input can be a 1D, 2D, 3D or higher-dimensional input for the Rastrigin function.
        A: The scale factor (typically set to 10).
        """
        if (type(X) != np.array):
            raise TypeError(f"input X must be a numpy array")


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
        # Beale function operates in a 2D space, so n_dimension is set to 2 by default
        super().__init__(n_dimension=n_dimension, 
                         search_space_ranges=[(-4.5, 4.5)] * n_dimension, 
                         global_minimum=0,
                         global_minimumX=[3, 0.5],  # Global minimum at (3, 0.5)
                         noises=noises, 
                         irrelevant_dims=irrelevant_dims,
                         description="""The Beale function is a benchmark function with many local minima near the global minimum at (3, 0.5). 
It is used to test optimization algorithms in 2D space. The typical search range is between -4.5 and 4.5 for both x and y.""")

    @property
    def _get_source_code(self, X):
        x, y = X[..., 0], X[..., 1]
        return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2

    def evaluate(self, X):
        """
        Evaluates the Beale function at a given input X.

        X: Can be a single point (1D array), multiple points (2D array), or a meshgrid input (3D for plotting).
        """
        total_dims = self.n_dimension + self.irrelevant_dims
        if X.shape[-1] != total_dims:
            raise ValueError(f"Input dimension mismatch: expected {total_dims} dimensions but got {X.shape[-1]} dimensions.")
        
        if X.ndim == 1:
            x, y = X 
            return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2

        elif X.ndim == 2:
            return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2
        
        elif X.ndim == 3:
            X_mesh, Y_mesh = X
            return (1.5 - X_mesh + X_mesh * Y_mesh)**2 + (2.25 - X_mesh + X_mesh * Y_mesh**2)**2 + (2.625 - X_mesh + X_mesh * Y_mesh**3)**2
        
        else:
            x, y = X[..., 0], X[..., 1]  # Unpack the first two dimensions
            return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2


class Sphere(BenchmarkFunctions):
    def __init__(self, n_dimension, noises=0.0, irrelevant_dims=0):
        """
        Initializes the Sphere function benchmark.
        The Sphere function is typically used as a basic benchmark function for optimization.
        It has a global minimum at 0, where f(0) = 0, and is defined for a search space range [-5.12, 5.12].
        """
        super().__init__(n_dimension=n_dimension, 
                         search_space_ranges=[(-5.12, 5.12)] * n_dimension, 
                         global_minimum=0,
                         global_minimumX=[0] * n_dimension,
                         noises=noises,
                         irrelevant_dims=irrelevant_dims,
                         description="""The Sphere function is a simple benchmark function for optimization. 
The global minimum is at (0, 0, ..., 0), where f(0) = 0. The typical search space for each dimension 
is [-5.12, 5.12], and the function is convex and unimodal.""")
        
    @property
    def _get_source_code(self, X):
        return np.sum(X**2, axis=-1)
    
    def evaluate(self, X):
        """
        Evaluates the Sphere function at a given point X.
        Supports single point evaluation (1D), multiple points (2D), and meshgrid input (3D).
        Handles higher-dimensional arrays (ndim > 3) as well.
        """
        
        total_dims = self.n_dimension + self.irrelevant_dims
        if X.shape[-1] != total_dims:
            raise ValueError(f"Input dimension mismatch: expected {total_dims} dimensions but got {X.shape[-1]} dimensions.")

        if X.ndim == 1:
            return np.sum(X**2)
        
        elif X.ndim == 2:
            return np.sum(X**2, axis=1)
        
        elif X.ndim == 3:
            X, Y = X  # Unpack meshgrid tuple
            return X**2 + Y**2
        
        else:
            return np.sum(X**2, axis=-1)



class BinaryTreeStructuredFunction(BenchmarkFunctions):

    def __init__(self, n_dimension = 7, noises = 0.0, irrelevant_dims=0):
        super().__init__(n_dimension, 
                         search_space_ranges = [(-5.0, 5.0)], 
                         global_minimum=0, 
                         global_minimumX=[0]*n_dimension, 
                         noises = noises, 
                         irrelevant_dims=irrelevant_dims, 
                         description="""Synthetic binary tree structured function, it has 7 relevant dimensions,
                        first three inputs must be binary, the last 4 inputs can take any values within search space """)

    def evaluate(self, X):
        if X[0] == 0:
            if X[1] == 0:
                val = X[3]**2 + 0.1 + 0.5
            else:
                val = X[4]**2 + 0.2 + 0.5
        else:
            if X[2] == 0:
                val = X[5]**2 + 0.3 + 0.9
            else:
                val = X[6]**2 + 0.4 + 0.9

        return val