import torch
from torch import Tensor
from botorch.utils.transforms import standardize
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.gp_regression import SingleTaskGP, ExactGP
from botorch.models.fully_bayesian import MaternKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.transforms import standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.types import _DefaultType, DEFAULT
from typing import Any, Optional, Union, Dict, Tuple, List
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.acquisition import UpperConfidenceBound, AcquisitionFunction, LogExpectedImprovement
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from tqdm import tqdm
from botorch.optim.optimize import optimize_acqf
from itertools import product

class MLP_BO_Optimiser:

    def __init__(self):
        self.params = None
        self.objective_func = None
        self.last_error = None

    def optimise(self, code_str, 
                    objective_function_name: str,
                    sample_per_batch=1,
                    n_iter=25, 
                    initial_points=5,):
        r"""
        Optimize the hyperparameters using Bayesian Optimization.
        :param code_str: A string defining the objective function.
        :param n_iter: Number of optimization iterations.
        :param initial_points: Number of initial random samples.
        """
        namespace = {}
        exec(code_str, namespace)
        self.objective_func = namespace.get(objective_function_name)
        if not callable(self.objective_func):
            raise ValueError("The code string must define a callable function")
        
        return self._run_bayesian_optimisation(
                                        n_iter=n_iter, 
                                        initial_points = initial_points,
                                        sample_per_batch= sample_per_batch)

    def _botorch_objective(self, x):
        """
        A thin wrapper to map input tensor to hyperparameters for MLP
        """
        np_params = x.detach().cpu().numpy().squeeze()
        params = {
            "learning_rate": np_params[0],
            "momentum": np_params[1],
            "weight_decay": np_params[2],
            "num_epochs": 50,
        }

        print("current X: ", params)

        return torch.tensor(self.objective_func(**params), dtype=torch.float64)

    def _normalize_to_unit_cube(self, data, bounds):
        lower_bounds = bounds[0].to(data.device)  # Move to the same device as `data`
        upper_bounds = bounds[1].to(data.device)
        return (data - lower_bounds) / (upper_bounds - lower_bounds)


    def _denormalize_from_unit_cube(self, data, bounds):
        lower_bounds = bounds[0].to(data.device)
        upper_bounds = bounds[1].to(data.device)
        return data * (upper_bounds - lower_bounds) + lower_bounds


    def _run_bayesian_optimisation(self, n_iter, initial_points, sample_per_batch):
        """
        Run Bayesian Optimisation for hyperparameter tuning
        """
        # Define bounds in normalized space [0, 1]
        bounds = torch.Tensor([[0, 0, 0], [1, 1, 0.1]])
        if torch.cuda.is_available():
            bounds = bounds.cuda()

        # Generate initial Sobol samples
        train_x = draw_sobol_samples(bounds=bounds, n=initial_points, q=sample_per_batch).squeeze(1).to(torch.float64)
        train_y = torch.tensor([self._botorch_objective(x).item() for x in train_x], dtype=torch.float64).view(-1, 1)

        if torch.cuda.is_available():
            train_x = train_x.cuda()
            train_y = train_y.cuda()

        # Define GP model and fit
        likelihood = GaussianLikelihood().to(train_x.device, dtype=torch.float64)
        gp = SingleTaskGP(train_X=train_x, train_Y=train_y, likelihood=likelihood).to(train_x.device, dtype=torch.float64)

        mll = ExactMarginalLogLikelihood(likelihood, gp)
        fit_gpytorch_mll_torch(mll)

        # Initialize acquisition function
        acq_function = LogExpectedImprovement(model=gp, best_f=train_y.max())

        best_candidate = None
        best_y = float('-inf')
        accuracies = []

        with tqdm(total=n_iter, desc="Bayesian Optimization Progress", unit="iter") as pbar:
            for i in range(n_iter):
                # Optimize acquisition function
                candidate, acq_value = optimize_acqf(
                    acq_function=acq_function,
                    bounds=bounds,
                    q=1,                # Optimize for one candidate
                    num_restarts=10,    # Number of random restarts
                    raw_samples=100,    # Number of raw samples for initialization
                )

                # Evaluate the objective at the new candidate
                new_y = self._botorch_objective(candidate).view(1, 1).to(train_y.device)
                new_y_value = new_y.item()

                if new_y_value > best_y:
                    best_y = new_y_value
                    best_candidate = candidate

                # Update training data
                train_x = torch.cat([train_x, candidate.view(1, -1)])

                train_y = train_y.view(-1, 1)
                train_y = torch.cat([train_y, new_y], dim=0)
                train_y = train_y.view(-1)

                # Update the GP model with the new data
                gp.set_train_data(inputs=train_x, targets=train_y, strict=False)

                # Update acquisition function with new best value
                acq_function = LogExpectedImprovement(model=gp, best_f=train_y.max())

                accuracies.append(new_y_value)
                pbar.set_postfix({"Best Y": best_y})
                pbar.update(1)

        return accuracies, best_y, best_candidate
