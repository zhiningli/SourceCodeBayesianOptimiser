import torch
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
from src.utils.kernels import TransformedOverlapKernel


# Example data
train_x = torch.randint(0, 4, (20, 3)).float()  # Shape: (20, 3)
train_y = torch.randn(20).unsqueeze(-1)  # Shape: (20, 1)


# Use the SingleTaskGP model with custom kernel
likelihood = GaussianLikelihood()
model = SingleTaskGP(train_x, train_y, covar_module=TransformedOverlapKernel(num_dimensions=train_x.size(-1)))

# Now you can use the model with BoTorch's acquisition functions
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

# Define the acquisition function
ucb = UpperConfidenceBound(model, beta=0.1)

# Define bounds for optimization
bounds = torch.tensor([[0] * train_x.size(-1), [3] * train_x.size(-1)], dtype=torch.float)
print("Shape of train_x:", train_x.shape)
print("Shape of train_y:", train_y.shape)


# Optimize the acquisition function
new_x, _ = optimize_acqf(
    acq_function=ucb,
    bounds=bounds,
    q=1,
    num_restarts=5,
    raw_samples=20,
)

print("Optimized input:", new_x)


