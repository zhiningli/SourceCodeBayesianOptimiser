import torch
from botorch.models import MixedSingleTaskGP
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.fit import fit_gpytorch_mll_torch
from src.data.data_models import SVMHyperParameterSpace
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
import traceback
import logging

class SVMSourceCodeOptimiser:

    def __init__(self):
        self.objective_func_hyperparameter_search_space = SVMHyperParameterSpace
        self.params = None
        self.objective_func = None
    
    def optimise(self, code_str, n_iter=20, initial_points=10, sample_per_batch=1):
        try:
            namespace = {}
            exec(code_str, namespace)
            # Retrieve the function dynamically
            self.objective_func = namespace.get("run_svm_classification")
            if not callable(self.objective_func):
                raise ValueError("The code string must define a callable 'run_svm_classification' function.")
            
            # Run Bayesian optimization
            self._run_bayesian_optimisation(n_iter=n_iter, initial_points=initial_points, sample_per_batch=sample_per_batch)
            return True
        except Exception as e:
            error_message = traceback.format_exc()
            logging.error("Execution failed with error: %s", error_message)
            self.last_error = error_message
            return False
        
    def _botorch_objective(self, x):
        """
        This method acts as the objective function for BoTorch, using the parameters to call `self.objective_func`.
        """
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

    def _run_bayesian_optimisation(self, n_iter, initial_points, sample_per_batch):
        # Define the bounds for kernel, C, coef0, and gamma search space
        bounds = torch.tensor([[0, 0.1, 0.0, 0], [len(SVMHyperParameterSpace["kernel"]["options"])-1, 10.0, 1.0, len(SVMHyperParameterSpace["gamma"]["options"])-1]], dtype=torch.float64)

        # Step 1: Initial Sample Points for Training
        train_x = draw_sobol_samples(bounds=bounds, n=initial_points, q=sample_per_batch).squeeze(1).to(dtype=torch.float64)
        train_x = (train_x - bounds[0]) / (bounds[1] - bounds[0])
        unnormalized_train_x = train_x * (bounds[1] - bounds[0]) + bounds[0]

        train_y = torch.tensor([self._botorch_objective(x) for x in unnormalized_train_x], dtype=torch.float64).unsqueeze(-1)
        train_y = standardize(train_y).view(-1, 1)

        # Initialise the GP model with double precision
        gp = MixedSingleTaskGP(train_x, train_y, cat_dims=[0, 3]).to(torch.float64)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(torch.float64)

        # Fit the GP model
        fit_gpytorch_mll_torch(mll)

        # Define the acquisition function
        ei = LogExpectedImprovement(model=gp, best_f=train_y.max())

        for i in range(n_iter):
            candidate, _ = optimize_acqf(
                acq_function=ei,
                bounds=bounds,
                q=sample_per_batch,
                num_restarts=5,
                raw_samples=20
            )

            train_y = train_y.view(-1, 1)
            new_y = self._botorch_objective(candidate).view(1, 1)
            train_x = torch.cat([train_x, candidate.view(1, -1).to(torch.float64)])
            train_y = torch.cat([train_y, new_y], dim=0)
            train_y = train_y.view(-1)
        
            gp.set_train_data(inputs=train_x, targets=train_y, strict=False)
            ei = LogExpectedImprovement(model=gp, best_f=train_y.max())
        
        best_idx = train_y.argmax()
        best_hyperparams = train_x[best_idx]
        best_kernel = SVMHyperParameterSpace["kernel"]["options"][int(best_hyperparams[0].item())]
        best_C = best_hyperparams[1].item()
        best_coef0 = best_hyperparams[2].item()
        best_gamma = SVMHyperParameterSpace["gamma"]["options"][int(best_hyperparams[3].item())]

        print("Best Hyperparameters:")
        print("Kernel:", best_kernel)
        print("C:", best_C)
        print("Coef0:", best_coef0)
        print("Gamma:", best_gamma)
        print("Accuracy:", self.objective_func(kernel=best_kernel, C=best_C, coef0=best_coef0, gamma=best_gamma))
