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
from botorch.optim.optimize import optimize_acqf_discrete
from itertools import product

class MLP_BO_Optimiser:

    def __init__(self):
        self.params = None
        self.objective_func = None
        self.last_error = None

    def optimise(self, code_str, 
                    search_space: Tensor,
                    sample_per_batch=1,
                    n_iter=25, 
                    initial_points=10,):
        r"""
        Optimize the hyperparameters using Bayesian Optimization.
        :param code_str: A string defining the objective function.
        :param n_iter: Number of optimization iterations.
        :param initial_points: Number of initial random samples.
        """

        namespace = {}
        exec(code_str, namespace)
        self.objective_func = namespace.get("run_mlp_classification")
        if not callable(self.objective_func):
            raise ValueError("The code string must define a callable function")
        
        return self._run_bayesian_optimisation(
                                        n_iter=n_iter,
                                        search_space = search_space, 
                                        initial_points = initial_points,
                                        sample_per_batch= sample_per_batch)

    def _botorch_objective(self, x):
        """
        A thin wrapper to map input tensor to hyperparameters for MLP
        """
        np_params = x.detach().cpu().numpy().squeeze()

        return torch.tensor(self.objective_func(**params), dtype=torch.float64)

    def _normalize_to_unit_cube(self, data, bounds):
        lower_bounds = bounds[0].to(data.device)  # Move to the same device as `data`
        upper_bounds = bounds[1].to(data.device)
        return (data - lower_bounds) / (upper_bounds - lower_bounds)


    def _denormalize_from_unit_cube(self, data, bounds):
        lower_bounds = bounds[0].to(data.device)
        upper_bounds = bounds[1].to(data.device)
        return data * (upper_bounds - lower_bounds) + lower_bounds


    def _run_bayesian_optimisation(self, 
                                    n_iter,
                                    search_space,
                                    initial_points,
                                    sample_per_batch,
                                   ):
        r"""
        Run Bayesian Optimisation for hyperparameter tuning
        """
        if torch.cuda.is_available():
            search_space = search_space.cuda()

        train_x = draw_sobol_samples(bounds=search_space, n=initial_points, q=sample_per_batch).squeeze(1).cuda()

        train_y = torch.tensor([self._botorch_objective(x).item() for x in train_x], dtype=torch.float64).view(-1, 1)

        normalised_train_x = self._normalize_to_unit_cube(train_x, search_space)

        choices = torch.tensor(list(product(*[range(len(search_space[dim])) for dim in search_space])), dtype=torch.float64)

        normalized_choices = (choices - torch.tensor([self.search_space[dim][0] for dim in search_space], dtype=torch.float64)) / \
                            torch.tensor([search_space[dim][-1] - search_space[dim][0] for dim in search_space], dtype=torch.float64)

        if torch.cuda.is_available():
            normalised_train_x = normalised_train_x.cuda()
            train_y = train_y.cuda()
            normalized_choices = normalized_choices.cuda()

        likelihood = GaussianLikelihood().to(torch.float64)
        gp = (ExactGP(
            train_X = normalised_train_x,
            train_Y= train_y,
            likelihood=likelihood,
        ).to(torch.float64))

        if torch.cuda.is_available():
            likelihood = likelihood.cuda()
            gp = gp.cuda()

        mll = ExactMarginalLogLikelihood(likelihood, gp).to(torch.float64)
        fit_gpytorch_mll_torch(mll)
        acq_function = UpperConfidenceBound(model = gp, beta = 2)

        best_candidate = None
        best_y = float('-inf')
        accuracies = []
        with tqdm(total=n_iter, desc="Bayesian Optimization Progress", unit="iter") as pbar:
            for i in range(n_iter):
                candidate, acq_value = optimize_acqf_discrete(
                    acq_function=acq_function,
                    q=1,
                    choices=normalized_choices, 
                    max_batch_size=2048,
                    unique=True
                )

                print(f"Raw Candidate: {candidate} with acquisition value {acq_value}")

                # Normalize candidate to evaluate y
                candidate = self._denormalize_from_unit_cube(candidate, search_space)

                if torch.cuda.is_available():
                    candidate = candidate.cuda()

                new_y = self._botorch_objective(candidate).view(1, 1)
                new_y = new_y.to(train_y.device)
                new_y_value = new_y.item()

                if new_y_value >= best_y:
                    best_y = new_y_value
                    best_candidate = candidate

                candidate = self._normalize_to_unit_cube(candidate, search_space)

                accuracies.append(new_y_value)

                train_x = torch.cat([train_x, candidate.view(1, -1)])
                train_y = train_y.view(-1, 1)
                train_y = torch.cat([train_y, new_y], dim=0)
                train_y = train_y.view(-1)

                gp.set_train_data(inputs=train_x, targets=train_y, strict=False)
                acq_function = UpperConfidenceBound(model = gp, beta = 2)

                pbar.set_postfix({"Best Y": best_y})
                pbar.update(1)
        return accuracies, best_y, best_candidate