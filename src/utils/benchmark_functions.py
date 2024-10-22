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

    def __init__(self, n_dimension, noises=0.0, irrelevant_dims = 0):
        super().__init__(n_dimension=n_dimension, 
                         search_space_range=(-5.12, 5.12), 
                         global_minimum=0,
                         global_minimumX=[0] * n_dimension,
                         noises = noises, 
                         irrelevant_dims = irrelevant_dims,
                         description="") 

    def evaluate(self, X, A=3):
        X = np.array(X)
        if X.ndim == 1:
            # Single point evaluation
            return A * len(X) + np.sum(X**2 - A * np.cos(2 * np.pi * X))
        elif X.ndim == 2:
            # Multiple points or meshgrid input
            return A * X.shape[1] + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)
        elif X.ndim == 3:
            # Meshgrid input (used for 3D plotting)
            X, Y = X
            return A * 2 + (X**2 - A * np.cos(2 * np.pi * X)) + (Y**2 - A * np.cos(2 * np.pi * Y))


class Beale(BenchmarkFunctions):
    def __init__(self, n_dimension=2):
        super().__init__(n_dimension=n_dimension, 
                         search_space_range=(-4.5, 4.5), 
                         global_minimum=0,
                         global_minimumX=(3, 0.5))

    def evaluate(self, X):
        """
        Evaluates the Beale function at a given point X.
        Supports single point evaluation, 2D array of points, or meshgrid input.
        """
        print(X)
        X = np.array(X)
        
        if X.ndim == 1:  # Single point evaluation
            x, y = X  # Unpack x and y
            return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2
        
        elif X.ndim == 2:  # Multiple points evaluation (X is a 2D array with shape (n_points, 2))
            x, y = X[:, 0], X[:, 1]
            return (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2
        
        elif X.ndim == 3:  # Meshgrid input for plotting (X is a meshgrid tuple like (X_mesh, Y_mesh))
            X, Y = X
            return (1.5 - X + X * Y)**2 + (2.25 - X + X * Y**2)**2 + (2.625 - X + X * Y**3)**2




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
