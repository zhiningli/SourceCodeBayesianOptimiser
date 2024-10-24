import numpy as np
import torch

from src.torchIntegration.GPytorchGP import GPModel
from src.torchIntegration.PItorch import PI
from src.torchIntegration.torchBayesianOptimiser import BayesianOptimiser
from src.utils.benchmark_functions.synthetic_functions import Rastrigin, Sphere
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import RBFKernel

n_dimensions = 2
benchmark = Sphere(n_dimension=n_dimensions)

likelihood = GaussianLikelihood()
gp_model = GPModel(torch.tensor([[0.0, 0.0]]), torch.tensor([0.0]), likelihood)
acquisition_func = PI(xi=0.1)


optimiser = BayesianOptimiser(acquisition_func, gp_model, n_iter=50, objective_func=benchmark.evaluate)

bounds = benchmark.search_space
lower_bounds = np.array([b[0] for b in bounds])
upper_bounds = np.array([b[1] for b in bounds])
X_init = np.random.uniform(lower_bounds, upper_bounds, (50, benchmark.n_dimension))
y_init = np.array([benchmark.evaluate(x) for x in X_init])

result = optimiser.optimise(X_init, y_init, bounds)
print(f"Global minimum found at: {result['best_point']}, with value: {result['best_value']}")