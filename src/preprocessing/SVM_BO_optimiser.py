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

class SVM_BO_Optimiser:


    def __init__(self):

        self.kernels = {}


    def _bo_objective(self):
        pass

    def optimise(self):
        pass