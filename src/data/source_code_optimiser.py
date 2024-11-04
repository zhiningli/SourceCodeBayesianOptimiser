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




class SVMSourceCodeOptimiser:

    def __init__(self):
        self.objective_func_hyperparameter_search_space = SVMHyperParameterSpace
        self.params = None
        self.objective_func = None
        
    def botorch_objective(self):
        """
        This method is a thin wrapper around the objective function to ensure the compatibility of objective function and 
        """
        np_params = self.params.detach().numpy().flatten()
        kernel_idx = int(np.round(np_params[0]))
        gamma_idx = int(np.round(np_params[3]))

        kernel = SVMHyperParameterSpace["kernel"]["options"][kernel_idx]
        C = np_params[1]
        coef0 = np_params[2]
        gamma = SVMHyperParameterSpace["gamma"]["options"][gamma_idx]

        result = self.objective_func(kernel = kernel, C = C, coef0 = coef0, gamma= gamma)
        return torch.tensor(result)
    
    def run_bayesian_optimisation(self, n_iter = 20, initial_points = 10, sample_per_batch = 1):
        # Define the bounds for kernel, C, coef0 and gamma search space
        bounds = torch.tensor([[0, 0.1, 0.0, 0], [len(SVMHyperParameterSpace["kernel"]["options"])-1, 10.0, 1.0, len(gamma = SVMHyperParameterSpace["gamma"]["options"])-1]])

        # Step 1: Initial Sample Points for Training
        train_x = draw_sobol_samples(bounds=bounds, n = initial_points, q = sample_per_batch).squeeze(1)
        train_x = (train_x - bounds[0]) / (bounds[1] - bounds[0])
        unnormalized_train_x = train_x * (bounds[1] - bounds[0]) + bounds[0]

        train_y = torch.tensor([self.botorch_objective(x) for x in unnormalized_train_x]).unsqueeze(-1)
        train_y = standardize(train_y).view(-1, 1)

        # Initialise the GP model
        gp = MixedSingleTaskGP(train_x, train_y, cat_dims=[0, 3])
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

        # Fit the GP model
        fit_gpytorch_mll_torch(mll)

        # Define the acquisition function
        ei = LogExpectedImprovement(model = gp, best_f = train_y.max())

        for i in range(n_iter):
            candidate, _ = optimize_acqf(
                acq_function=ei,
                bounds=bounds,
                q = sample_per_batch,
                num_restarts=5,
                raw_samples=20
            )

            train_y = train_y.view(-1, 1)
            new_y = self.botorch_objective(candidate).view(1, 1)
            train_x = torch.cat([train_x, candidate.view(1, -1)])
            train_y = torch.cat([train_y, new_y], dim=0)
            train_y = train_y.view(-1)
        
            gp.set_train_data(inputs=train_x, targets=train_y, strict=False)

            ei = LogExpectedImprovement(model=gp, best_f=train_y.max())
        
        best_idx = train_y.argmax()
        best_hyperparams = train_x[best_idx]
        best_kernel = SVMHyperParameterSpace["kernel"]["options"][int(best_hyperparams[0].items())]
        best_C = best_hyperparams[1].item()
        best_coef0 = best_hyperparams[2].item()
        best_gamma = SVMHyperParameterSpace["gamma"]["options"][int(best_hyperparams[3].item())]
        print("Best Hyperparameters:")
        print("Kernel:", best_kernel)
        print("C:", best_C)
        print("Coef0:", best_coef0)
        print("Gamma:", best_gamma)
        print("Accuracy:", run_svm_classification(best_kernel, best_C, best_gamma, best_coef0))
