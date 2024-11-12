from gpytorch.kernels import ScaleKernel, HammingIMQKernel, RBFKernel
from gpytorch.priors import NormalPrior
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultitaskMultivariateNormal
from botorch.acquisition import ExpectedImprovement
import torch

class SVM_Expected_Improvement:

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

    def __init__(self, train_X, train_y, likelihood, kernel):
        super(SVM_GP_model, self).__init__(train_X, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultitaskMultivariateNormal(mean_x, covar_x)


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
        """Combine the individual kernels into one composite kernel and return the complete SVM_BO_Kernel object."""
        self._SVM_BO_Kernel.kernel = (
            self._SVM_BO_Kernel.kernel_for_svm_kernel *
            self._SVM_BO_Kernel.kernel_for_svm_C *
            self._SVM_BO_Kernel.kernel_for_svm_gamma *
            self._SVM_BO_Kernel.kernel_for_svm_coef0
        )
        return self._SVM_BO_Kernel
