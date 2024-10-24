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