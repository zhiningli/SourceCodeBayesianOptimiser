import logging
import traceback
from math import sqrt
from typing import Any, Optional, Union, Dict, Tuple, List

import numpy as np
import torch
from torch import Tensor

from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood, Likelihood

from botorch.optim import optimize_acqf_discrete
from botorch.models import SingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.fully_bayesian import MaternKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.transforms import standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.types import _DefaultType, DEFAULT
from botorch.acquisition import LogExpectedImprovement, AcquisitionFunction
from botorch.optim.fit import fit_gpytorch_mll_torch

from src.data.data_models import SVMHyperParameterSpace


class SVM_BO_optimiser:

    def __init__(self):
        self.objective_func_hyperparameter_search_space = SVMHyperParameterSpace
        self.params = None
        self.objective_func = None
    
    def optimise(self, code_str, n_iter=20, initial_points=10, sample_per_batch=1,
                                                    svm_kernel_lengthscale_prior_mean = 1,
                                                    svm_kernel_outputscale_prior_mean = 1,
                                                    svm_C_lengthscale_prior_mean = 1.5,
                                                    svm_C_outputscale_prior_mean = 1.5,
                                                    svm_gamma_lengthscale_prior_mean = 1,
                                                    svm_gamma_outputscale_prior_mean = 1,
                                                    svm_coef0_lengthscale_prior_mean = 1.5,
                                                   svm_coef0_outputscale_prior_mean = 1
                ):
        try:
            namespace = {}
            exec(code_str, namespace)
            self.objective_func = namespace.get("run_svm_classification")
            if not callable(self.objective_func):
                raise ValueError("The code string must define a callable 'run_svm_classification' function.")
            
            return self._run_bayesian_optimisation(n_iter=n_iter, initial_points=initial_points, sample_per_batch=sample_per_batch,
                                                    svm_kernel_lengthscale_prior_mean = svm_kernel_lengthscale_prior_mean,
                                                    svm_kernel_outputscale_prior_mean = svm_kernel_outputscale_prior_mean,
                                                    svm_C_lengthscale_prior_mean = svm_C_lengthscale_prior_mean,
                                                    svm_C_outputscale_prior_mean = svm_C_outputscale_prior_mean,
                                                    svm_gamma_lengthscale_prior_mean = svm_gamma_lengthscale_prior_mean,
                                                    svm_gamma_outputscale_prior_mean = svm_gamma_outputscale_prior_mean,
                                                    svm_coef0_lengthscale_prior_mean = svm_coef0_lengthscale_prior_mean,
                                                    svm_coef0_outputscale_prior_mean = svm_coef0_outputscale_prior_mean)
        except Exception as e:
            error_message = traceback.format_exc()
            logging.error("Execution failed with error: %s", error_message)
            self.last_error = error_message
            return False
        
    def _botorch_objective(self, x):
        np_params = x.detach().numpy().flatten()
        kernel_idx = int(np.round(np_params[0]))
        gamma_idx = int(np.round(np_params[3]))

        kernel = SVMHyperParameterSpace["kernel"]["options"][kernel_idx]
        C = np_params[1]
        coef0 = np_params[2]
        gamma = SVMHyperParameterSpace["gamma"]["options"][gamma_idx]

        result = self.objective_func(kernel=kernel, C=C, coef0=coef0, gamma=gamma)
        return torch.tensor(result, dtype=torch.float64)  # Ensure the result tensor is also double precision

    def _run_bayesian_optimisation(self, 
                                   n_iter, 
                                   initial_points, 
                                   sample_per_batch,
                                   svm_kernel_lengthscale_prior_mean,
                                   svm_kernel_outputscale_prior_mean,
                                   svm_C_lengthscale_prior_mean,
                                   svm_C_outputscale_prior_mean,
                                   svm_gamma_lengthscale_prior_mean,
                                   svm_gamma_outputscale_prior_mean,
                                   svm_coef0_lengthscale_prior_mean,
                                   svm_coef0_outputscale_prior_mean):
        # Define the bounds for kernel, C, coef0, and gamma search space
        bounds = torch.tensor([[0, 0.1, 0.0, 0], [len(SVMHyperParameterSpace["kernel"]["options"])-1, 10.0, 1.0, len(SVMHyperParameterSpace["gamma"]["options"])-1]], dtype=torch.float64)
        
        # Constrain the kernel and gamma search space to discrete values)
        discrete_dims = [0, 3]
        discrete_values = {
            0: [0, 1, 2, 3],  # Allowed values for dim 0
            3: [0, 1]  # Allowed values for dim 3
        }

        # Step 1: Initial Sample Points for Training
        train_x = draw_sobol_samples(bounds=bounds, n=initial_points, q=sample_per_batch).squeeze(1).to(dtype=torch.float64)
        train_x = (train_x - bounds[0]) / (bounds[1] - bounds[0])
        unnormalized_train_x = train_x * (bounds[1] - bounds[0]) + bounds[0]

        train_y = torch.tensor([self._botorch_objective(x) for x in unnormalized_train_x], dtype=torch.float64).unsqueeze(-1)
        train_y = standardize(train_y).view(-1, 1)

        likelihood = GaussianLikelihood().to(torch.float64)

        gp = (SVM_GP_model(train_X=train_x, train_Y=train_y, likelihood=likelihood, 
                                svm_kernel_lengthscale_prior_mean = svm_kernel_lengthscale_prior_mean ,
                                svm_kernel_outputscale_prior_mean = svm_kernel_outputscale_prior_mean,
                                svm_C_lengthscale_prior_mean = svm_C_lengthscale_prior_mean,
                                svm_C_outputscale_prior_mean = svm_C_outputscale_prior_mean,
                                svm_gamma_lengthscale_prior_mean = svm_gamma_lengthscale_prior_mean,
                                svm_gamma_outputscale_prior_mean = svm_gamma_outputscale_prior_mean,
                                svm_coef0_lengthscale_prior_mean = svm_coef0_lengthscale_prior_mean,
                                svm_coef0_outputscale_prior_mean = svm_coef0_outputscale_prior_mean
                          ).to(torch.float64))

        mll = ExactMarginalLogLikelihood(likelihood, gp).to(torch.float64)

        fit_gpytorch_mll_torch(mll)

        ei = LogExpectedImprovement(model=gp, best_f=train_y.max())
        best_candidate = None
        best_y = float('-inf')
        accuracies = []

        for i in range(n_iter):
            # Generate initial conditions for this iteration (e.g., Sobol sampling or custom initialization)
            initial_conditions = draw_sobol_samples(bounds=bounds, n=1, q=sample_per_batch).squeeze(1).to(dtype=torch.float64)
            
            # Generate one candidate using your custom function
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
            ei = LogExpectedImprovement(model=gp, best_f=train_y.max())

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



class SVM_Expected_Improvement(LogExpectedImprovement):

    def __init__(self, model, best_f):
        super().__init__(model, best_f)
    
    def _constraints(self, x):
        # Ensure conditions are checked using the same tensor type (BoolTensor)
        condition1 = (x[:, 0] >= 0).bool()
        condition2 = (x[:, 0] <= 4).bool()
        
        # Safely combine the conditions with bitwise AND
        return torch.all(condition1 & condition2)

    def forward(self, x):
        # Apply rounding to x without in-place modification to avoid breaking gradients
        x = x.clone()  # Clone to avoid in-place modification issues
        x[:, 0] = torch.round(x[:, 0])

        if not self._constraints(x):
            return super().forward(float("-inf"))
        
        return super().forward(x)



class SVM_GP_model(SingleTaskGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        svm_kernel_lengthscale_prior_mean: float,
        svm_kernel_outputscale_prior_mean: float,
        svm_C_lengthscale_prior_mean: float,
        svm_C_outputscale_prior_mean: float,
        svm_gamma_lengthscale_prior_mean: float,
        svm_gamma_outputscale_prior_mean: float,
        svm_coef0_lengthscale_prior_mean: float,
        svm_coef0_outputscale_prior_mean: float,
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


# class SVM_GP_model_Legacy(ExactGP):

#     def __init__(self, train_X, train_y, likelihood, kernel, input_transform: Optional[InputTransform] = None):
#         super(SVM_GP_model_Legacy, self).__init__(train_X, train_y, likelihood)
#         self.mean_module = ConstantMean()
#         self.covar_module = kernel
#         self.input_transfom = input_transform

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return MultivariateNormal(mean_x, covar_x)


# class SVM_BO_Kernel_Legacy:
#     def __init__(self):
#         self.kernel_for_svm_kernel = None
#         self.kernel_for_svm_C = None
#         self.kernel_for_svm_gamma = None
#         self.kernel_for_svm_coef0 = None
#         self.kernel = None

#     @classmethod
#     def builder(cls):
#         return SVM_BO_Kernel_builder()

#     def forward(self, x1, x2=None, **params):
#         if self.kernel is None:
#             raise ValueError("Kernel has not been built. Use the builder to construct the kernel before calling.")
#         return self.kernel(x1, x2, **params)

#     def __call__(self, x1, x2=None, **params):
#         return self.forward(x1, x2, **params)

# class SVM_BO_Kernel_builder:
#     def __init__(self):
#         self._SVM_BO_Kernel = SVM_BO_Kernel_Legacy()

#     def build_kernel_for_svm_kernel(self, outputscale_prior_mean):
#         r"""
#         Constructs a HammingIMQKernel for the SVM kernel.

#         Parameters:
#         outputscale_prior_mean: float
#             The prior mean for the output scale for HammingIMQKernel that represents the SVM kernel hyperparameter.
#         """
#         outputscale_prior = NormalPrior(outputscale_prior_mean, 2.0)

#         self._SVM_BO_Kernel.kernel_for_svm_kernel = ScaleKernel(
#             HammingIMQKernel(vocab_size=4),
#             outputscale_prior=outputscale_prior
#         )
#         return self

#     def build_kernel_for_svm_C(self, lengthscale_prior_mean, outputscale_prior_mean):
#         r"""
#         Constructs an RBFKernel for the SVM hyperparameter C.

#         Parameters:
#         lengthscale_prior_mean: float
#             The prior mean for the lengthscale of the RBF kernel.
#         outputscale_prior_mean: float
#             The prior mean for the output scale of the RBF kernel.
#         """
#         lengthscale_prior = NormalPrior(lengthscale_prior_mean, 2.0)
#         outputscale_prior = NormalPrior(outputscale_prior_mean, 2.0)

#         self._SVM_BO_Kernel.kernel_for_svm_C = ScaleKernel(
#             RBFKernel(lengthscale_prior=lengthscale_prior),
#             outputscale_prior=outputscale_prior
#         )
#         return self

#     def build_kernel_for_svm_gamma(self, outputscale_prior_mean):
#         r"""
#         Constructs a HammingIMQKernel for the SVM gamma hyperparameter.

#         Parameters:
#         outputscale_prior_mean: float
#             The prior mean for the output scale for the HammingIMQKernel representing the SVM gamma.
#         """
#         outputscale_prior = NormalPrior(outputscale_prior_mean, 2.0)

#         self._SVM_BO_Kernel.kernel_for_svm_gamma = ScaleKernel(
#             HammingIMQKernel(vocab_size=2),
#             outputscale_prior=outputscale_prior
#         )
#         return self

#     def build_kernel_for_svm_coef0(self, lengthscale_prior_mean, outputscale_prior_mean):
#         r"""
#         Constructs an RBFKernel for the SVM hyperparameter coef0.

#         Parameters:
#         lengthscale_prior_mean: float
#             The prior mean for the lengthscale of the RBF kernel.
#         outputscale_prior_mean: float
#             The prior mean for the output scale of the RBF kernel.
#         """
#         lengthscale_prior = NormalPrior(lengthscale_prior_mean, 2.0)
#         outputscale_prior = NormalPrior(outputscale_prior_mean, 2.0)

#         self._SVM_BO_Kernel.kernel_for_svm_coef0 = ScaleKernel(
#             RBFKernel(lengthscale_prior=lengthscale_prior),
#             outputscale_prior=outputscale_prior
#         )
#         return self

#     def build(self) -> SVM_BO_Kernel_Legacy:
#         """Combine the individual kernels using ProductKernel and return the complete SVM_BO_Kernel object."""
#         self._SVM_BO_Kernel.kernel = ProductKernel(
#             self._SVM_BO_Kernel.kernel_for_svm_kernel,
#             self._SVM_BO_Kernel.kernel_for_svm_C,
#             self._SVM_BO_Kernel.kernel_for_svm_gamma,
#             self._SVM_BO_Kernel.kernel_for_svm_coef0
#         )
#         return self._SVM_BO_Kernel
