import numpy as np
import torch
import gpytorch

class BayesianOptimiser:
    def __init__(self, acquisition_func, model, n_iter, objective_func):
        self.acquisition_func = acquisition_func
        self.model = model
        self.n_iter = n_iter
        self.objective_func = objective_func

    def optimise(self, X_init, y_init, bounds, grid_density=50):
        X = X_init
        y = y_init

        # Train the GP model
        self._train_gp_model(X, y)

        for i in range(self.n_iter):
            X_candidates = self._generate_candidates(bounds, grid_density)
            acquisition_values = self.acquisition_func.compute(X_candidates, self.model, y.min())

            best_candidate_idx = np.argmax(acquisition_values)
            next_point = X_candidates[best_candidate_idx].numpy()  # Convert torch.Tensor to numpy.ndarray

            # Evaluate the objective function at the next candidate point
            next_value = self.objective_func(next_point)

            print(f"Iteration {i + 1}: Next point: {next_point}, Next value: {next_value}")

            # Update the dataset
            X = np.vstack([X, next_point])
            y = np.append(y, next_value)

            # Retrain the model with the new data
            self._train_gp_model(X, y)

        best_idx = np.argmin(y)
        return {"best_point": X[best_idx], "best_value": y[best_idx]}

    def _train_gp_model(self, X_train, y_train):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)

        self.model.set_train_data(inputs=X_train, targets=y_train, strict=False)

        self.model.train()
        self.model.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        for _ in range(100):  # Training loop
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

    def _generate_candidates(self, bounds, grid_density):
        lower_bounds = np.array([b[0] for b in bounds])
        upper_bounds = np.array([b[1] for b in bounds])

        # Generate uniform samples across the search space
        X_candidates = np.random.uniform(lower_bounds, upper_bounds, (grid_density, len(bounds)))
        return torch.tensor(X_candidates, dtype=torch.float32)
