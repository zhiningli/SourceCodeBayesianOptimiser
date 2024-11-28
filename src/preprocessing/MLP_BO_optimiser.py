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


from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood, Likelihood


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

    def optimise(self, code_str, n_iter=20, initial_points=10):
        r"""
        Optimize the hyperparameters using Bayesian Optimization.
        :param code_str: A string defining the objective function.
        :param n_iter: Number of optimization iterations.
        :param initial_points: Number of initial random samples.
        """
        try:
            namespace = {}
            exec(code_str, namespace)
            self.objective_func = namespace.get("run_mpl_classitication")
            if not callable(self.objective_func):
                raise ValueError("The code string must define a callable function")
            
            return self._run_bayesian_optimisation(n_iter=n_iter, initial_points = initial_points)
        except Exception as e:
            error_message = traceback.format_exc()
            logging.error("Execution failed with error: %s", error_message)
            self.last_error = error_message
            return False
    
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


    def _run_bayesian_optimisation(self, n_iter_initial_points):
        r"""
        Run Bayesian Optimisation for hyperparameter tuning
        """

        bounds = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0], 
            [3, 3, 3, 3, 2, 2, 2]   
        ], dtype=torch.float())

        train_x = torch.rand((n_iter_initial_points, bounds.size(1))) * (bounds[1] - bounds[0]) + bounds[0]
        train_y = torch.tensor([self._botorch_objective(x).items for x in train_x])
        train_y = standardize(train_y)

        likelihood = GaussianLikelihood().to(torch.float64)


class MLP_GP_model(SingleTaskGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        MLP_hidden1_nu_prior_mean: float,
        MLP_hidden2_nu_prior_mean: float,
        MLP_hidden3_nu_prior_mean: float,
        likelihood: Optional[Likelihood],
        train_Yvar: Optional[Tensor] = None,
        outcome_transform: Optional[Union[OutcomeTransform, _DefaultType]] = DEFAULT,
        input_transform: Optional[InputTransform] = None,
    ) -> None:

        cat_kernel_for_gamma = ScaleKernel(
            CategoricalKernel(
                ard_num_dims=1,
                lengthscale_prior=LogNormalPrior(loc=svm_gamma_lengthscale_prior_mean, scale=sqrt(3.0)),
                lengthscale_constraint=GreaterThan(1e-06)
            ),
            outputscale_prior=LogNormalPrior(loc=svm_gamma_outputscale_prior_mean, scale=sqrt(3.0)),
            outputscale_constraint=GreaterThan(0.1)
        )

        cat_kernel_for_kernel = ScaleKernel(
            CategoricalKernel(
                ard_num_dims=1,
                lengthscale_prior=LogNormalPrior(loc=svm_kernel_lengthscale_prior_mean, scale=sqrt(3.0)),
                lengthscale_constraint=GreaterThan(1e-06)
            ),
            outputscale_prior=LogNormalPrior(loc=svm_kernel_outputscale_prior_mean, scale=sqrt(3.0)),
            outputscale_constraint=GreaterThan(0.1)
        )

        continuous_kernel_for_C = ScaleKernel(
            MaternKernel(
                ard_num_dims=1,
                lengthscale_prior=LogNormalPrior(loc=svm_C_lengthscale_prior_mean, scale=sqrt(3.0)),
                lengthscale_constraint=GreaterThan(1e-06)
            ),
            outputscale_prior=LogNormalPrior(loc=svm_C_outputscale_prior_mean, scale=sqrt(3.0)),
            outputscale_constraint=GreaterThan(0.1)
        )

        continuous_kernel_for_coef0 = ScaleKernel(
            RBFKernel(
                ard_num_dims=1,
                lengthscale_prior=LogNormalPrior(loc=svm_coef0_lengthscale_prior_mean, scale=sqrt(3.0)),
                lengthscale_constraint=GreaterThan(1e-06)
            ),
            outputscale_prior=LogNormalPrior(loc=svm_coef0_outputscale_prior_mean, scale=sqrt(3.0)),
            outputscale_constraint=GreaterThan(0.1)
        )

        covar_module = cat_kernel_for_kernel * continuous_kernel_for_C * continuous_kernel_for_coef0 * cat_kernel_for_gamma
        print("Gamma Kernel Lengthscale Prior: LogNormalPrior with loc =", cat_kernel_for_gamma.base_kernel.lengthscale_prior.loc.item(),
            "and scale =", cat_kernel_for_gamma.base_kernel.lengthscale_prior.scale.item())
        print("Gamma Kernel Outputscale Prior: LogNormalPrior with loc =", cat_kernel_for_gamma.outputscale_prior.loc.item(),
            "and scale =", cat_kernel_for_gamma.outputscale_prior.scale.item())

        print("Kernel Type Kernel Lengthscale Prior: LogNormalPrior with loc =", cat_kernel_for_kernel.base_kernel.lengthscale_prior.loc.item(),
            "and scale =", cat_kernel_for_kernel.base_kernel.lengthscale_prior.scale.item())
        print("Kernel Type Kernel Outputscale Prior: LogNormalPrior with loc =", cat_kernel_for_kernel.outputscale_prior.loc.item(),
            "and scale =", cat_kernel_for_kernel.outputscale_prior.scale.item())

        print("C Kernel Lengthscale Prior: LogNormalPrior with loc =", continuous_kernel_for_C.base_kernel.lengthscale_prior.loc.item(),
            "and scale =", continuous_kernel_for_C.base_kernel.lengthscale_prior.scale.item())
        print("C Kernel Outputscale Prior: LogNormalPrior with loc =", continuous_kernel_for_C.outputscale_prior.loc.item(),
            "and scale =", continuous_kernel_for_C.outputscale_prior.scale.item())

        print("Coef0 Kernel Lengthscale Prior: LogNormalPrior with loc =", continuous_kernel_for_coef0.base_kernel.lengthscale_prior.loc.item(),
            "and scale =", continuous_kernel_for_coef0.base_kernel.lengthscale_prior.scale.item())
        print("Coef0 Kernel Outputscale Prior: LogNormalPrior with loc =", continuous_kernel_for_coef0.outputscale_prior.loc.item(),
            "and scale =", continuous_kernel_for_coef0.outputscale_prior.scale.item())

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

