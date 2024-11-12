from gpytorch.kernels import ScaleKernel, HammingIMQKernel, RBFKernel, ProductKernel
from gpytorch.priors import NormalPrior
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal

from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.fit import fit_gpytorch_mll_torch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models.transforms.input import InputTransform

from src.data.data_models import SVMHyperParameterSpace

from typing import Optional
import numpy as np
import torch
import traceback
import logging

class SVM_BO_optimiser:

    def __init__(self):
        self.objective_func_hyperparameter_search_space = SVMHyperParameterSpace
        self.params = None
        self.objective_func = None
    
    def optimise(self, code_str, n_iter=20, initial_points=10, sample_per_batch=1,
                                                    svm_kernel_outputscale_prior_mean = 1,
                                                    svm_C_lengthscale_prior_mean = 1.5,
                                                    svm_C_outputscale_prior_mean = 1.5,
                                                    svm_gamma_outputscale_prior_mean = 1,
                                                    svm_coef0_lengthscale_prior_mean = 1.5,
                                                    svm_coef0_outputscale_prior_mean = 1):
        try:
            namespace = {}
            exec(code_str, namespace)
            # Retrieve the function dynamically
            self.objective_func = namespace.get("run_svm_classification")
            if not callable(self.objective_func):
                raise ValueError("The code string must define a callable 'run_svm_classification' function.")
            
            # Run Bayesian optimization
            return self._run_bayesian_optimisation(n_iter=n_iter, initial_points=initial_points, sample_per_batch=sample_per_batch,
                                                    svm_kernel_outputscale_prior_mean = svm_kernel_outputscale_prior_mean,
                                                    svm_C_lengthscale_prior_mean = svm_C_lengthscale_prior_mean,
                                                    svm_C_outputscale_prior_mean = svm_C_outputscale_prior_mean,
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

        # Call the dynamically assigned function with parameters
        result = self.objective_func(kernel=kernel, C=C, coef0=coef0, gamma=gamma)
        return torch.tensor(result, dtype=torch.float64)  # Ensure the result tensor is also double precision

    def _run_bayesian_optimisation(self, 
                                   n_iter, 
                                   initial_points, 
                                   sample_per_batch,
                                   svm_kernel_outputscale_prior_mean,
                                   svm_C_lengthscale_prior_mean,
                                   svm_C_outputscale_prior_mean,
                                   svm_gamma_outputscale_prior_mean,
                                   svm_coef0_lengthscale_prior_mean,
                                   svm_coef0_outputscale_prior_mean):
        # Define the bounds for kernel, C, coef0, and gamma search space
        bounds = torch.tensor([[0, 0.1, 0.0, 0], [len(SVMHyperParameterSpace["kernel"]["options"])-1, 10.0, 1.0, len(SVMHyperParameterSpace["gamma"]["options"])-1]], dtype=torch.float64)

        # Step 1: Initial Sample Points for Training
        train_x = draw_sobol_samples(bounds=bounds, n=initial_points, q=sample_per_batch).squeeze(1).to(dtype=torch.float64)
        train_x = (train_x - bounds[0]) / (bounds[1] - bounds[0])
        unnormalized_train_x = train_x * (bounds[1] - bounds[0]) + bounds[0]

        train_y = torch.tensor([self._botorch_objective(x) for x in unnormalized_train_x], dtype=torch.float64).unsqueeze(-1)
        train_y = standardize(train_y).view(-1, 1)

        kernel = (
            SVM_BO_Kernel.builder()
            .build_kernel_for_svm_kernel(outputscale_prior_mean=svm_kernel_outputscale_prior_mean)
            .build_kernel_for_svm_C(lengthscale_prior_mean=svm_C_lengthscale_prior_mean, outputscale_prior_mean=svm_C_outputscale_prior_mean)
            .build_kernel_for_svm_gamma(outputscale_prior_mean=svm_gamma_outputscale_prior_mean)
            .build_kernel_for_svm_coef0(lengthscale_prior_mean=svm_coef0_lengthscale_prior_mean, outputscale_prior_mean=svm_coef0_outputscale_prior_mean)
            .build()
        )

        # Initialise the GP model with double precision
        likelihood = GaussianLikelihood().to(torch.float64)

        gp = SVM_GP_model(train_X=train_x, train_y=train_y, likelihood=likelihood, kernel=kernel).to(torch.float64)

        mll = ExactMarginalLogLikelihood(likelihood, gp).to(torch.float64)

        # Fit the GP model
        fit_gpytorch_mll_torch(mll)

        ei = SVM_Expected_Improvement(model=gp, best_f=train_y.max())

        candidates = []
        for i in range(n_iter):
            candidate, _ = optimize_acqf(
                acq_function=ei,
                bounds=bounds,
                q=sample_per_batch,
                num_restarts=5,
                raw_samples=20
            )

            candidates.append(candidate)

            train_y = train_y.view(-1, 1)   
            # Evaluate the candidate and ensure new_y has shape [1, 1]
            new_y = self._botorch_objective(candidate).view(1, 1)  # Ensure new_y shape [1, 1]
            # Concatenate the new candidate and new_y to training data
            train_x = torch.cat([train_x, candidate.view(1, -1)])  # Ensure train_x matches shape
            train_y = torch.cat([train_y, new_y], dim=0)  # Concatenate along dimension 0
            train_y = train_y.view(-1)

            gp.set_train_data(inputs=train_x, targets=train_y, strict=False)
            ei = SVM_Expected_Improvement(model=gp, best_f=train_y.max())

        return candidates



class SVM_Expected_Improvement(LogExpectedImprovement):

    def __init__(self, model, best_f, constraints):
        super().__init__(model, best_f)
        self.constraints = constraints
    
    def _constraints(self, x):
        return torch.all((x[:, 0] >= 0) & x[:, 0] <= 4)

    def forward(self, x):
        x[:, 0] = torch.round(x[:, 0])

        if not self.constraints(x):
            return super().forward(float("-inf"))


class SVM_GP_model(ExactGP):

    def __init__(self, train_X, train_y, likelihood, kernel, input_transform: Optional[InputTransform] = None):
        super(SVM_GP_model, self).__init__(train_X, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = kernel
        self.input_transfom = input_transform

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def transform_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_transfom is not None:
            return self.input_transform(x)
        return x

class SVM_BO_Kernel:
    def __init__(self):
        self.kernel_for_svm_kernel = None
        self.kernel_for_svm_C = None
        self.kernel_for_svm_gamma = None
        self.kernel_for_svm_coef0 = None
        self.kernel = None

    @classmethod
    def builder(cls):
        return SVM_BO_Kernel_builder()

    def forward(self, x1, x2=None, **params):
        if self.kernel is None:
            raise ValueError("Kernel has not been built. Use the builder to construct the kernel before calling.")
        return self.kernel(x1, x2, **params)

    def __call__(self, x1, x2=None, **params):
        return self.forward(x1, x2, **params)

class SVM_BO_Kernel_builder:
    def __init__(self):
        self._SVM_BO_Kernel = SVM_BO_Kernel()

    def build_kernel_for_svm_kernel(self, outputscale_prior_mean):
        r"""
        Constructs a HammingIMQKernel for the SVM kernel.

        Parameters:
        outputscale_prior_mean: float
            The prior mean for the output scale for HammingIMQKernel that represents the SVM kernel hyperparameter.
        """
        outputscale_prior = NormalPrior(outputscale_prior_mean, 2.0)

        self._SVM_BO_Kernel.kernel_for_svm_kernel = ScaleKernel(
            HammingIMQKernel(vocab_size=4),
            outputscale_prior=outputscale_prior
        )
        return self

    def build_kernel_for_svm_C(self, lengthscale_prior_mean, outputscale_prior_mean):
        r"""
        Constructs an RBFKernel for the SVM hyperparameter C.

        Parameters:
        lengthscale_prior_mean: float
            The prior mean for the lengthscale of the RBF kernel.
        outputscale_prior_mean: float
            The prior mean for the output scale of the RBF kernel.
        """
        lengthscale_prior = NormalPrior(lengthscale_prior_mean, 2.0)
        outputscale_prior = NormalPrior(outputscale_prior_mean, 2.0)

        self._SVM_BO_Kernel.kernel_for_svm_C = ScaleKernel(
            RBFKernel(lengthscale_prior=lengthscale_prior),
            outputscale_prior=outputscale_prior
        )
        return self

    def build_kernel_for_svm_gamma(self, outputscale_prior_mean):
        r"""
        Constructs a HammingIMQKernel for the SVM gamma hyperparameter.

        Parameters:
        outputscale_prior_mean: float
            The prior mean for the output scale for the HammingIMQKernel representing the SVM gamma.
        """
        outputscale_prior = NormalPrior(outputscale_prior_mean, 2.0)

        self._SVM_BO_Kernel.kernel_for_svm_gamma = ScaleKernel(
            HammingIMQKernel(vocab_size=2),
            outputscale_prior=outputscale_prior
        )
        return self

    def build_kernel_for_svm_coef0(self, lengthscale_prior_mean, outputscale_prior_mean):
        r"""
        Constructs an RBFKernel for the SVM hyperparameter coef0.

        Parameters:
        lengthscale_prior_mean: float
            The prior mean for the lengthscale of the RBF kernel.
        outputscale_prior_mean: float
            The prior mean for the output scale of the RBF kernel.
        """
        lengthscale_prior = NormalPrior(lengthscale_prior_mean, 2.0)
        outputscale_prior = NormalPrior(outputscale_prior_mean, 2.0)

        self._SVM_BO_Kernel.kernel_for_svm_coef0 = ScaleKernel(
            RBFKernel(lengthscale_prior=lengthscale_prior),
            outputscale_prior=outputscale_prior
        )
        return self

    def build(self) -> SVM_BO_Kernel:
        """Combine the individual kernels using ProductKernel and return the complete SVM_BO_Kernel object."""
        self._SVM_BO_Kernel.kernel = ProductKernel(
            self._SVM_BO_Kernel.kernel_for_svm_kernel,
            self._SVM_BO_Kernel.kernel_for_svm_C,
            self._SVM_BO_Kernel.kernel_for_svm_gamma,
            self._SVM_BO_Kernel.kernel_for_svm_coef0
        )
        return self._SVM_BO_Kernel
