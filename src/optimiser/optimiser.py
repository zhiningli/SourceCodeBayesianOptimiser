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
            X_candidates = self._generate_candidates(X, y, bounds)

            # Compute the acquisition value using the acquisition function provided.
            acquisition_values = self.acquisition.compute(X_candidates, self.model)

            # Logging acquisition values for debugging
            print(f"Iteration {i + 1}: Max acquisition value is {np.max(acquisition_values)} at candidate {X_candidates[np.argmax(acquisition_values)]}")

            # Select the candidate with the highest acquisition value
            best_candidate_idx = np.argmax(acquisition_values)
            next_point = X_candidates[best_candidate_idx]
            print(type(next_point))
            next_value = self._evaluate_objective(next_point)
            # print("next_value: ", next_value)
            # print(next_point)
            # X, Y = next_point

            # print("calculating it again: ", (1.5 - X + X * Y)**2 + (2.25 - X + X * Y**2)**2 + (2.625 - X + X * Y**3)**2)

            print(f"Next point selected: {next_point} with value: {next_value}")

            # Update training data with the new point
            X = np.vstack([X, next_point])
            y = np.append(y, next_value)
            self.model.fit(X, y)

            # Logging progress
            best_value = np.min(y)

        best_idx = np.argmin(y)
        return {"best_point": X[best_idx], "best_value": y[best_idx]}

    def _generate_candidates(self, X, y, bounds, n_total=600):
        """
        Generates candidate points with a mix of uniform sampling across the space
        and sampling concentrated around the current best point, ensuring all points
        fall within the specified bounds.

        Parameters:
        X (np.ndarray): Training data points.
        y (np.ndarray): Training data values.
        bounds (list of tuples): Bounds for each dimension.
        n_total (int): Total number of candidate points to generate.

        Returns:
        np.ndarray: An array of candidate points.
        """
        n_uniform = n_total // 3  # Number of uniformly sampled points
        n_near_best = n_total - n_uniform  # Number of points near the best candidate

        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])
        uniform_samples = np.random.uniform(lower_bounds, upper_bounds, (n_uniform, len(bounds)))
        # Generate uniform random samples within the bounds

        # Identify the best point from the training data
        best_idx = np.argmin(y)
        best_point = X[best_idx]

        # Generate near-best samples by adding Gaussian noise around the best point
        noise_scale = 0.05 * np.abs(np.array([b[1] - b[0] for b in bounds]))
        near_best_samples = np.array([np.random.normal(best_point[i], noise_scale[i], n_near_best) for i in range(len(bounds))]).T
        print("Best point is: ", best_point)
        # Combine the uniform and near-best samples
        candidates = np.vstack([uniform_samples, near_best_samples])

        # Ensure that all generated candidate points fall within the bounds
        candidates = np.clip(candidates, lower_bounds, upper_bounds)

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
