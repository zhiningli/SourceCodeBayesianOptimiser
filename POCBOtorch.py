import numpy as np
import torch
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from botorch.models import MixedSingleTaskGP
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.fit import fit_gpytorch_mll_torch

from sklearn import datasets
data = datasets.load_iris()
X, y = data.data, data.target

# Corrected kernel options to avoid duplicate keys
kernel_options = {"linear": 0, "poly": 1, "rbf": 2, "sigmoid": 3}
gamma_options = {"scale": 0, "auto": 1}

def svm_objective(params):
    kernel = list(kernel_options.keys())[int(params[0].item())]
    C = params[1].item()
    coef0 = params[2].item()
    gamma = list(gamma_options.keys())[int(params[3].item())]  # Fixed typo here

    model = SVC(kernel=kernel, C=C, coef0=coef0, gamma=gamma)
    scores = cross_val_score(model, X, y, cv=5)

    return np.mean(scores)

def botorch_objective(params):
    np_params = params.detach().numpy()
    result = svm_objective(np_params)
    return torch.tensor(result)

# Step 1: Initial Sample Points for Training
train_x = draw_sobol_samples(bounds=torch.tensor([[0, 0.1, 0.0, 0], [3, 10.0, 1.0, 1]]), n=10, q=1).squeeze(1)
train_y = torch.tensor([botorch_objective(x) for x in train_x]).unsqueeze(-1)
train_y = standardize(train_y)

# Step 2: Initialize the GP Model
gp = MixedSingleTaskGP(train_x, train_y, cat_dims=[0, 3])
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)

# Fit the GP model
fit_gpytorch_mll_torch(mll)

# Step 3: Define the Acquisition Function
ei = ExpectedImprovement(model=gp, best_f=train_y.max())

# Step 4: Optimization Loop
bounds = torch.tensor([[0, 0.1, 0.0, 0], [3, 10.0, 1.0, 1]])

for i in range(20):  # Number of BO iterations
    # Optimize the acquisition function to find the next candidate point
    candidate, _ = optimize_acqf(
        acq_function=ei,
        bounds=bounds,
        q=1,
        num_restarts=5,
        raw_samples=20,
    )

    # Evaluate the candidate
    new_y = botorch_objective(candidate)
    new_y_std = standardize(new_y.unsqueeze(-1))
    
    # Update training data and retrain GP
    train_x = torch.cat([train_x, candidate])
    train_y = torch.cat([train_y, new_y_std.unsqueeze(-1)])
    gp.set_train_data(inputs=train_x, targets=train_y, strict=False)
    
    # Refit the GP model
    fit_gpytorch_mll(mll)

    # Update acquisition function
    ei = ExpectedImprovement(model=gp, best_f=train_y.max())

# Best Hyperparameters
best_idx = train_y.argmax()
best_hyperparams = train_x[best_idx]

# Decoding best parameters
best_kernel = list(kernel_options.keys())[int(best_hyperparams[0].item())]
best_C = best_hyperparams[1].item()
best_coef0 = best_hyperparams[2].item()
best_gamma = list(gamma_options.keys())[int(best_hyperparams[3].item())]

print("Best Hyperparameters:")
print("Kernel:", best_kernel)
print("C:", best_C)
print("Coef0:", best_coef0)
print("Gamma:", best_gamma)
print("Accuracy:", svm_objective(best_hyperparams.numpy()))
