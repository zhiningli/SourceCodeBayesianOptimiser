import numpy as np

class BenchmarkFunctions:
    def __init__(self, n_dimension, search_space_range, global_minimum, global_minimumX):
        self.n_dimension = n_dimension
        self._search_space = search_space_range 
        self._global_minimum = global_minimum
        self._global_minimumX = global_minimumX

    @property
    def search_space(self):
        """
        Return the recommended search space for the benchmark function.
        """
        return [self._search_space] * self.n_dimension
    
    @property
    def global_minimum(self):
        """
        Return the known global minimum value.
        """
        return self._global_minimum
    
    @property
    def global_minimumX(self):
        """
        Return the known global minimum location.
        """
        return [self._global_minimumX] * self.n_dimension
    

class Rastrigin(BenchmarkFunctions):

    def __init__(self, n_dimension):
        super().__init__(n_dimension=n_dimension, 
                         search_space_range=(-5.12, 5.12), 
                         global_minimum=0,
                         global_minimumX=[0] * n_dimension)  # Global minimum is a vector of zeros

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
