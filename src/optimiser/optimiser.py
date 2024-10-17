import numpy as np
from src.base import Model, Acquisition

class Optimiser:
    def __init__(self, acquisition: Acquisition, model: Model, n_iter, objective_func):
        self.acquisition = acquisition
        self.model = model
        self.n_iter = n_iter
        self.objective_func = objective_func

    def optimise(self, X, y, bounds, grid_density=50):
        """
        Optimizes the given objective function using Bayesian Optimization.

        Parameters:
        X (array-like): Initial training points.
        y (array-like): Initial objective values at training points.
        bounds (list of tuples): The bounds for each dimension of the input space.
        grid_density (int): Number of points per dimension for the grid search.

        Returns:
        dict: The best point and its corresponding value.
        """
        self.model.fit(X, y)

        for i in range(self.n_iter):
            # Generate candidate points using the custom strategy
            X_candidates = self._generate_candidates(X, y, bounds, n_total=200)

            # Compute acquisition values (PI) for the candidates
            acquisition_values = self.acquisition.compute(X_candidates, self.model)

            # Logging acquisition values for debugging
            print(f"Iteration {i + 1}: Max acquisition value is {np.max(acquisition_values)} at candidate {X_candidates[np.argmax(acquisition_values)]}")

            # Select the candidate with the highest acquisition value
            best_candidate_idx = np.argmax(acquisition_values)
            next_point = X_candidates[best_candidate_idx]
            next_value = self._evaluate_objective(next_point)

            print(f"Next point selected: {next_point} with value: {next_value}")

            # Update training data with the new point
            X = np.vstack([X, next_point])
            y = np.append(y, next_value)
            self.model.fit(X, y)

            # Logging progress
            best_value = np.min(y)
            print(f"Iteration {i + 1}: Best value so far is {best_value}")

        best_idx = np.argmin(y)
        return {"best_point": X[best_idx], "best_value": y[best_idx]}

    def _generate_candidates(self, X, y, bounds, n_total=200):
        """
        Generates candidate points with a mix of uniform sampling across the space
        and sampling concentrated around the current best point.

        Parameters:
        X (np.ndarray): Training data points.
        y (np.ndarray): Training data values.
        bounds (list of tuples): Bounds for each dimension.
        n_total (int): Total number of candidate points to generate.

        Returns:
        np.ndarray: An array of candidate points.
        """
        n_uniform = n_total // 2  # Number of uniformly sampled points
        n_near_best = n_total - n_uniform  # Number of points near the best candidate

        uniform_samples = np.array([np.random.uniform(b[0], b[1], n_uniform) for b in bounds]).T

        best_idx = np.argmin(y)
        best_point = X[best_idx]

        noise_scale = 0.20 * np.abs(np.array([b[1] - b[0] for b in bounds]))
        near_best_samples = np.array([np.random.normal(best_point[i], noise_scale[i], n_near_best) for i in range(len(bounds))]).T

        candidates = np.vstack([uniform_samples, near_best_samples])

        print(f"Generated {n_uniform} uniform samples and {n_near_best} samples near the best point {best_point}.")
        return candidates

    def _evaluate_objective(self, X):
        """
        Evaluates the objective function at a given point X.

        Parameters:
        X (np.ndarray): The point at which to evaluate the objective.

        Returns:
        float: The objective function value at the given point.
        """
        return self.objective_func(X)
