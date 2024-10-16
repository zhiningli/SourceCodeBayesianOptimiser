from src.base import Model, Acquisition, Kernel
import numpy as np

class Optimiser:

    def __init__(self, acquisition: Acquisition, model: Model, n_iter, objective_func):
        self.acquisition = acquisition
        self.model = model
        self.n_iter = n_iter
        self.objective_func = objective_func

    def optimise(self, X, y, bounds):
        """
        Optimizes the given objective function using Bayesian Optimization.

        Parameters:
        X (array-like): Initial training points.
        y (array-like): Initial objective values at training points.
        bounds (list of tuples): The bounds for each dimension of the input space.

        Returns:
        dict: The best point and its corresponding value.
        """
        self.model.fit(X, y)
        X = np.array(X)
        y = np.array(y)

        for i in range(self.n_iter):
            X_candidates = self._generate_candidates(bounds)

            ei_values = self.acquisition.compute(X_candidates, self.model)
            
            if ei_values.shape[0] != X_candidates.shape[0]:
                raise ValueError("Mismatch between the number of candidate points and EI values")

            next_point = X_candidates[np.argmax(ei_values)]
            next_value = self._evaluate_objective(next_point)

            X = np.vstack([X, next_point])
            y = np.append(y, next_value)

            self.model.fit(X, y)

            print(f"Iteration {i + 1}: Best value so far is {np.max(y)}")

        best_idx = np.argmax(y)
        return {"best_point": X[best_idx], "best_value": y[best_idx]}


    def _generate_candidates(self, bounds, num_candidates=100):
        """
        Generates random candidate points within the specified bounds.
        """
        dim = len(bounds)
        return np.random.uniform(
            low=[b[0] for b in bounds],
            high=[b[1] for b in bounds],
            size=(num_candidates, dim)
        )

    def _evaluate_objective(self, X):
        
        print("Evaluating the objective function", self.objective_func(X))
        return self.objective_func(X)
 
