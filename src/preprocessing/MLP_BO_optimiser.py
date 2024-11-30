import logging
import traceback
import torch
from torch import Tensor
from torch import nn
from botorch.utils.transforms import standardize
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from botorch.models import SingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.fully_bayesian import MaternKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.transforms import standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.types import _DefaultType, DEFAULT
from typing import Any, Optional, Union, Dict, Tuple, List
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.acquisition import UpperConfidenceBound, AcquisitionFunction
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from tqdm import tqdm


class MLP_BO_Optimiser:

    def __init__(self):
        self.params = None
        self.objective_func = None
        self.search_space = {
            'hidden1': [32, 64, 128],
            'hidden2': [32, 64, 128],
            'hidden3': [32, 64, 128],
            'hidden4': [32, 64, 128],
            'activation': ['ReLU','Tanh','LeakyReLu'],
            'lr': [0.0001, 0.001, 0.01],
            'weight_decay': [0.0, 0.0001, 0.001]
        }
        self.last_error = None

    def optimise(self, code_str, 
                MLP_hidden1_nu: float,
                MLP_hidden2_nu: float,
                MLP_hidden3_nu: float,
                MLP_hidden4_nu: float,
                MLP_lr_nu: float,
                MLP_activation_nu: float,
                MLP_weight_decay_nu: float,
                sample_per_batch=1,
                n_iter=20, 
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
        
        return self._run_bayesian_optimisation(n_iter=n_iter, 
                                                initial_points = initial_points,
                                                MLP_hidden1_nu = MLP_hidden1_nu,
                                                MLP_hidden2_nu = MLP_hidden2_nu,
                                                MLP_hidden3_nu = MLP_hidden3_nu,
                                                MLP_hidden4_nu = MLP_hidden4_nu,
                                                MLP_lr_nu = MLP_lr_nu,
                                                MLP_activation_nu = MLP_activation_nu,
                                                MLP_weight_decay_nu = MLP_weight_decay_nu,
            sample_per_batch=1,)

    def _botorch_objective(self, x):
        """
        A thin wrapper to map input tensor to hyperparameters for MLP
        """
        np_params = x.detach().numpy()
        params = {
            'hidden1': self.search_space['hidden1'][int(np_params[0])],
            'hidden2': self.search_space['hidden2'][int(np_params[1])],
            'hidden3': self.search_space['hidden3'][int(np_params[2])],
            'hidden4': self.search_space['hidden4'][int(np_params[3])],
            'activation': self.search_space['activation'][int(np_params[4])],
            'lr': self.search_space['lr'][int(np_params[5])],
            'weight_decay': self.search_space['weight_decay'][int(np_params[6])],
        }

        return torch.tensor(self.objective_func(**params), dtype=torch.float)


    def _run_bayesian_optimisation(self, 
                                    n_iter,
                                    initial_points,
                                    MLP_hidden1_nu: float,
                                    MLP_hidden2_nu: float,
                                    MLP_hidden3_nu: float,
                                    MLP_hidden4_nu: float,
                                    MLP_lr_nu: float,
                                    MLP_activation_nu: float,
                                    MLP_weight_decay_nu: float,
                                    sample_per_batch=1
                                   ):
        r"""
        Run Bayesian Optimisation for hyperparameter tuning
        """

        bounds = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0], 
            [3, 3, 3, 3, 2, 2, 2]   
        ], dtype=torch.float)

        discrete_dims = [0, 1, 2, 3, 4, 5, 6, 7]
        discrete_values = {
            0: [0, 1, 2],
            1: [0, 1, 2],
            2: [0, 1, 2],
            3: [0, 1, 2],
            4: [0, 1, 2],
            5: [0, 1, 2],
            6: [0, 1, 2],
        }

        train_x = torch.rand((initial_points, bounds.size(1))) * (bounds[1] - bounds[0]) + bounds[0]
        train_y = torch.tensor([self._botorch_objective(x).item for x in train_x])
        train_y = standardize(train_y)

        likelihood = GaussianLikelihood().to(torch.float64)

        gp = (MLP_GP_model(MLP_hidden1_nu = MLP_hidden1_nu, 
                           MLP_hidden2_nu = MLP_hidden2_nu,
                           MLP_hidden3_nu = MLP_hidden3_nu, 
                           MLP_hidden4_nu = MLP_hidden4_nu,
                           MLP_lr_nu = MLP_lr_nu,
                           MLP_activation_nu = MLP_activation_nu,
                           MLP_weight_decay_nu = MLP_weight_decay_nu,
                           train_X=train_x, train_Y=train_y, likelihood=likelihood
                          ).to(torch.float64))
        

        mll = ExactMarginalLogLikelihood(likelihood, gp).to(torch.float64)
        fit_gpytorch_mll_torch(mll)

        ei = UpperConfidenceBound(model=gp, best_f=train_y.max())
        best_candidate = None
        best_y = float('-inf')
        accuracies = []

        with tqdm(total=n_iter, desc="Bayesian Optimization Progress", unit="iter") as pbar:
            for i in range(n_iter):
                initial_conditions = draw_sobol_samples(bounds=bounds, n=1, q=sample_per_batch).squeeze(1).to(dtype=torch.float64)
                
                candidate, acq_value = self._optimize_acqf_with_discrete_search_space(
                    initial_conditions=initial_conditions,
                    acquisition_function=ei,
                    bounds=bounds,
                    discrete_dims=discrete_dims,
                    discrete_values=discrete_values
                )
                
                train_y = train_y.view(-1, 1)
                new_y = self._botorch_objective(candidate).view(1, 1)
                new_y_value = new_y.item()
                accuracies.append(new_y_value)
                if new_y_value >= best_y:
                    best_y = new_y_value 
                    best_candidate = candidate 
                train_x = torch.cat([train_x, candidate.view(1, -1)])
                train_y = torch.cat([train_y, new_y], dim=0)
                train_y = train_y.view(-1)

                gp.set_train_data(inputs=train_x, targets=train_y, strict=False)
                ei = UpperConfidenceBound(model=gp, best_f=train_y.max())

                # Update progress bar
                pbar.set_postfix({"Best Y": best_y})
                pbar.update(1)

        return accuracies, best_y, best_candidate
    
    def _optimize_acqf_with_discrete_search_space(
        self,
        initial_conditions: torch.Tensor,
        acquisition_function: AcquisitionFunction,
        bounds: torch.Tensor,
        discrete_dims: List[int],
        discrete_values: Dict[int, List[float]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lower_bounds = bounds[0]
        upper_bounds = bounds[1]

        candidates = []
        acq_values = []
        for i in range(initial_conditions.size(0)):
            candidate = initial_conditions[i].clone()

            # Enforce discrete constraints on specified dimensions
            for dim in discrete_dims:
                candidate[dim] = min(discrete_values[dim], key=lambda x: abs(x - candidate[dim]))

            # Enforce bounds by clipping the values
            candidate = torch.max(candidate, lower_bounds)
            candidate = torch.min(candidate, upper_bounds)

            # Evaluate the acquisition function
            acq_value = acquisition_function(candidate.unsqueeze(0))
            candidates.append(candidate)
            acq_values.append(acq_value)

        # Convert lists to tensors
        candidates = torch.stack(candidates)
        acq_values = torch.stack(acq_values)

        return candidates, acq_values


class MLP_GP_model(SingleTaskGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        MLP_hidden1_nu: float,
        MLP_hidden2_nu: float,
        MLP_hidden3_nu: float,
        MLP_hidden4_nu: float,
        MLP_lr_nu: float,
        MLP_activation_nu: float,
        MLP_weight_decay_nu: float,
        likelihood: Optional[Likelihood],
        train_Yvar: Optional[Tensor] = None,
        outcome_transform: Optional[Union[OutcomeTransform, _DefaultType]] = DEFAULT,
        input_transform: Optional[InputTransform] = None,
    ) -> None:

        matern_kernel_for_hidden1 = ScaleKernel(
            MaternKernel(
                nu = MLP_hidden1_nu,
            )
        )

        matern_kernel_for_hidden2 = ScaleKernel(
            MaternKernel(
                nu = MLP_hidden2_nu,
            )
        )

        matern_kernel_for_hidden3 = ScaleKernel(
            MaternKernel(
                nu = MLP_hidden3_nu,
            )
        )

        matern_kernel_for_hidden4 = ScaleKernel(
            MaternKernel(
                nu = MLP_hidden4_nu,
            )
        )

        matern_kernel_for_lr = ScaleKernel(
            MaternKernel(
                nu = MLP_lr_nu,
            )
        )

        matern_kernel_for_activation = ScaleKernel(
            MaternKernel(
                nu = MLP_activation_nu,
            )
        )

        matern_kernel_for_weight_decay = ScaleKernel(
            MaternKernel(
                nu = MLP_weight_decay_nu
            )
        )

        covar_module = matern_kernel_for_hidden1 * matern_kernel_for_hidden2 * matern_kernel_for_hidden3 * matern_kernel_for_hidden4 * matern_kernel_for_lr * matern_kernel_for_activation * matern_kernel_for_weight_decay

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

