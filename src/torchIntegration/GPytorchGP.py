import torch
import gpytorch
from torch import Tensor
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal


class GPModel(ExactGP):

    def __init__(self, train_x: Tensor, train_y: Tensor, likelihood: gpytorch.likelihoods.Likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()

        # Use ScaleKernel with RBFKernel (this will learn a scaling factor during optimization)
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# class GP:

#     def __init__(self, noise=1e-2):
#         self.noise = noise
#         self.model = None
#         self.likelihood = None
#         self.X_train = None
#         self.y_train = None

#     def fit(self, X, y):
#         self.X_train = torch.tensor(X, dtype=torch.float32)
#         self.y_train = torch.tensor(y, dtype=torch.float32)
    
#         self.likelihood = GaussianLikelihood()
#         self.model = GPModel(self.X_train, self.y_train, self.likelihood)

#         self.model.train()
#         self.likelihood.train()

#         optimiser = torch.optim.Adam(self.model.parameters(), lr=0.1)
#         mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

#         for i in range(100):
#             optimiser.zero_grad()
#             output = self.model.forward(self.X_train)
#             loss = -mll(output, self.y_train)
#             loss.backward()
#             optimiser.step()

#     def predict(self, X):
#         self.model.eval()
#         self.likelihood.eval()

#         X_test = torch.tensor(X, dtype=torch.float32)

#         with torch.no_grad(), gpytorch.settings.fast_pred_var():
#             pred_dist = self.likelihood(self.model(X_test))
#             mean = pred_dist.mean
#             std = pred_dist.variance.sqrt()
        
#         return mean.numpy(), std.numpy()

        