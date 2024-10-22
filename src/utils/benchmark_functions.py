import numpy as np

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
        pass


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
    
    def evaluate(self, X, A=10):
        """
        Evaluates the Rastrigin function for a given input X.

        X: Input can be a 1D, 2D, 3D or higher-dimensional input for the Rastrigin function.
        A: The scale factor (typically set to 10).
        """
        X = np.array(X)
        total_dims = self.n_dimension + self.irrelevant_dims

        # Error handling: Check if input dimensions match expected total dimensions
        if X.shape[-1] != total_dims:
            raise ValueError(f"Input dimension mismatch: expected {total_dims} dimensions but got {X.shape[-1]} dimensions.")
        
        # Handling 1D input (single point evaluation)
        if X.ndim == 1:
            return A * len(X) + np.sum(X**2 - A * np.cos(2 * np.pi * X))
        
        # Handling 2D input (multiple points)
        elif X.ndim == 2:
            return A * X.shape[1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)
        
        # Handling 3D input (used for plotting purposes, e.g., meshgrid for surface plots)
        elif X.ndim == 3:
            X, Y = X
            return A * 2 + (X**2 - A * np.cos(2 * np.pi * X)) + (Y**2 - A * np.cos(2 * np.pi * Y))
        
        # Handling ND input (N > 3)
        else:
            # This will handle higher-dimensional arrays (X.ndim > 3)
            return A * X.shape[-1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=-1)



import numpy as np

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

    def evaluate(self, X):
        """
        Evaluates the Beale function at a given input X.

        X: Can be a single point (1D array), multiple points (2D array), or a meshgrid input (3D for plotting).
        """
        X = np.array(X)
        total_dims = self.n_dimension + self.irrelevant_dims

        # Error handling: Check if input dimensions match expected total dimensions
        if X.shape[-1] != total_dims:
            raise ValueError(f"Input dimension mismatch: expected {total_dims} dimensions but got {X.shape[-1]} dimensions.")
        
        # Single point evaluation (1D)
        if X.ndim == 1:
            x, y = X  # Unpack x and y
            return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2
        
        # Multiple points evaluation (2D)
        elif X.ndim == 2:
            x, y = X[:, 0], X[:, 1]  # Unpack x and y for all points
            return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2
        
        # Meshgrid input for plotting (3D)
        elif X.ndim == 3:
            X_mesh, Y_mesh = X
            return (1.5 - X_mesh + X_mesh * Y_mesh)**2 + (2.25 - X_mesh + X_mesh * Y_mesh**2)**2 + (2.625 - X_mesh + X_mesh * Y_mesh**3)**2
        
        # Handling ND input (N > 3)
        else:
            # Unpack the first two dimensions (x and y) and ignore irrelevant dimensions
            x, y = X[..., 0], X[..., 1]  # Unpack the first two dimensions
            return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2




class Sphere(BenchmarkFunctions):
    def __init__(self, n_dimension):
        super().__init__(n_dimension=n_dimension, 
                         search_space_range=(-5.12, 5.12), 
                         global_minimum=0,
                         global_minimumX=0)

    def evaluate(self, X):
        X = np.array(X)
        if X.ndim == 1:
            return np.sum(X**2)
        elif X.ndim == 2:
            return np.sum(X**2, axis=1)
        elif X.ndim == 3:
            X, Y = X
            return X**2 + Y**2
