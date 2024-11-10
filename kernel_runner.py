import torch
from botorch.models import SingleTaskGP
from gpytorch.likelihoods import GaussianLikelihood
from src.utils.kernels import TransformedOverlapKernel
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.kernels import LinearKernel

# Example data
train_x = torch.randint(0, 4, (20, 3)).float() # Shape: (20, 3)
train_x = (train_x - train_x.min()) / (train_x.max() - train_x.min())
train_x = train_x.double()
train_y = torch.randn(20).unsqueeze(-1).double()  # Shape: (20, 1)

print("train_x shape: ", train_x.shape)
print("train_y_shape: ", train_y.shape)

linearKernel = LinearKernel()
transformedOverlapKernel = TransformedOverlapKernel()
likelihood = GaussianLikelihood()
model = SingleTaskGP(train_x, train_y, covar_module=transformedOverlapKernel)

ucb = UpperConfidenceBound(model, beta=0.1)

# Define bounds for optimization
bounds = torch.tensor([[0] * train_x.size(-1), [3] * train_x.size(-1)], dtype=torch.float)

# Optimize the acquisition function
new_x, _ = optimize_acqf(
    acq_function=ucb,
    bounds=bounds,
    q=1,
    num_restarts=5,
    raw_samples=20,
)

print("Optimized input:", new_x)


