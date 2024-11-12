import torch
from gpytorch.kernels import ScaleKernel, HammingIMQKernel, RBFKernel
from botorch.models import MixedSingleTaskGP
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.fit import fit_gpytorch_mll_torch
from src.data.data_models import SVMHyperParameterSpace
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
import numpy as np

class SVM_BO_Kernel:

    def __init__(self):
        self.kernelPriors = {
            "outputscale_prior" : {
                "kernel": None, 
                "C": None,
                "coef0": None,
                "gamma": None
            },
            "lengthscale_prior": {
                "C": None,
                "coef0": None,
            }
        }
        self.kernels = None
        self._construct()

    def set_prior(self, priorCategory: str, object: str, prior):
        self.kernelPriors[priorCategory][object] = prior

    @classmethod
    def builder(self):
        return SVM_BO_Kernel_builder()

    def _bo_objective(self, **kwargs):
        pass

    def _construct(self):
        self.kernel_for_svm_kernel = ScaleKernel(HammingIMQKernel(vocab_size=4), outputscale_prior = self.kernelPriors["outputscale_prior"]["kernel"])
        self.kernel_for_svm_C = ScaleKernel(RBFKernel(lengthscale_prior=self.kernelPriors["lengthscale_prior"]["C"]), outputscale_prior=self.kernelPriors["outputscale_prior"]["C"])            
        self.kernel_for_svm_coef0 = ScaleKernel(RBFKernel(lengthscale_prior=self.kernelPriors["lengthscale_prior"]["coef0"]), outputscale_prior=self.kernelPriors["outputscale_prior"]["coef0"])
        self.kernel_for_svm_gamma = ScaleKernel(HammingIMQKernel(vocab_size=2), outputscale_prior=self.kernelPriors["outputscale_prior"]["gamma"])
        self.kernels = self.kernel_for_svm_kernel * self.kernel_for_svm_C * self.kernel_for_svm_coef0 * self.kernel_for_svm_gamma



class SVM_BO_Kernel_builder:

    def __init__(self):
        self._SVM_G