import logging
import traceback
import torch
from torch import Tensor
from torch import nn
from botorch.utils.transforms import standardize
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from botorch.models import SingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.fully_bayesian import MaternKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.transforms import standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.types import _DefaultType, DEFAULT
from typing import Any, Optional, Union, Dict, Tuple, List
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.acquisition import UpperConfidenceBound, AcquisitionFunction
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.priors import LogNormalPrior
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from tqdm import tqdm


class MLP_BO_Optimiser:

    def __init__(self):
        self.params = None
        self.objective_func = None
        self.search_space = {
            'conv_feature_num': [8, 16, 32],
            'conv_kernel_size': [3, 5, 7],
            'conv_stride': [1, 2],
            'hidden1': [128, 512, 1024],
            'hidden2': [128, 512, 1024],
            'activation': ['ReLU','Tanh','LeakyReLU'],
            'lr': [0.0001, 0.001, 0.01],
            'weight_decay': [0.0, 0.0001, 0.001],
            'epoch': [5, 10, 15],
            'batch_size': [64, 128]
        }
        self.last_error = None

    def optimise(self, code_str, 
                MLP_conv_feature_num_nu: float,
                MLP_conv_kernel_size_nu: float,
                MLP_conv_stride_nu: float,
                MLP_hidden1_nu: float,
                MLP_hidden2_nu: float,
                MLP_lr_nu: float,
                MLP_activation_nu: float,
                MLP_weight_decay_nu: float,
                MLP_epoch_nu: float,
                MLP_batch_size_nu: float,
                sample_per_batch=1,
                n_iter=20, 
                initial_points=10,):
        r"""
        Optimize the hyperparameters using Bayesian Optimization.
        :param code_str: A string defining the objective function.
        :param n_iter: Number of optimization iterations.
        :param initial_points: Number of initial random samples.
        """

        namespace = {}
        exec(code_str, namespace)
        self.objective_func = namespace.get("run_mlp_classification")
        if not callable(self.objective_func):
            raise ValueError("The code string must define a callable function")
        
        return self._run_bayesian_optimisation(n_iter=n_iter, 
                                                initial_points = initial_points,
                                                MLP_conv_feature_num_nu = MLP_conv_feature_num_nu,
                                                MLP_conv_kernel_size_nu = MLP_conv_kernel_size_nu,
                                                MLP_conv_stride_nu= MLP_conv_stride_nu,
                                                MLP_hidden1_nu = MLP_hidden1_nu,
                                                MLP_hidden2_nu = MLP_hidden2_nu,
                                                MLP_lr_nu = MLP_lr_nu,
                                                MLP_activation_nu = MLP_activation_nu,
                                                MLP_weight_decay_nu = MLP_weight_decay_nu,
                                                MLP_epoch_nu = MLP_epoch_nu,
                                                MLP_batch_size_nu = MLP_batch_size_nu,
                                                sample_per_batch= sample_per_batch)

    def _botorch_objective(self, x):
        """
        A thin wrapper to map input tensor to hyperparameters for MLP
        """
        np_params = x.detach().numpy().squeeze()
        params = {
            'conv_feature_num': self.search_space['conv_feature_num'][int(np_params[0])],
            'conv_kernel_size': self.search_space['conv_kernel_size'][int(np_params[1])],
            'conv_stride': self.search_space['conv_stride'][int(np_params[2])],
            'hidden1': self.search_space['hidden1'][int(np_params[3])],
            'hidden2': self.search_space['hidden2'][int(np_params[4])],
            'lr': self.search_space['lr'][int(np_params[5])],
            'activation': self.search_space['activation'][int(np_params[6])],
            'weight_decay': self.search_space['weight_decay'][int(np_params[7])],
            'epoch': self.search_space['epoch'][int(np_params[8])],
            'batch_size': self.search_space['batch_size'][int(np_params[9])],
        }

        return torch.tensor(self.objective_func(**params), dtype=torch.float)


    def _run_bayesian_optimisation(self, 
                                    n_iter,
                                    initial_points,
                                    MLP_conv_feature_num_nu : float,
                                    MLP_conv_kernel_size_nu : float,
                                    MLP_conv_stride_nu : float,
                                    MLP_hidden1_nu: float,
                                    MLP_hidden2_nu: float,
                                    MLP_lr_nu: float,
                                    MLP_activation_nu: float,
                                    MLP_weight_decay_nu: float,
                                    MLP_epoch_nu : float,
                                    MLP_batch_size_nu : float,
                                    sample_per_batch: float
                                   ):
        r"""
        Run Bayesian Optimisation for hyperparameter tuning
        """

        bounds = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [3, 3, 2, 3, 3, 3, 3, 3, 3, 2]   
        ], dtype=torch.float)

        discrete_dims = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        discrete_values = {
            0: [0, 1, 2],
            1: [0, 1, 2],
            2: [0, 1],
            3: [0, 1, 2],
            4: [0, 1, 2],
            5: [0, 1, 2],
            6: [0, 1, 2],
            7: [0, 1, 2],
            8: [0, 1, 2],
            9: [0, 1],
        }
        print("Running bayesian optimisation...")
        train_x = torch.rand((initial_points, bounds.size(1))) * (bounds[1] - bounds[0]) + bounds[0]
        print("Running objective function for 10 initial points...")
        train_y = torch.tensor([self._botorch_objective(x).item() for x in train_x], dtype=torch.float).view(-1, 1)
        likelihood = GaussianLikelihood().to(torch.float64)
        gp = (MLP_GP_model( MLP_conv_feature_num_nu = MLP_conv_feature_num_nu,
                           MLP_conv_kernel_size_nu = MLP_conv_kernel_size_nu,
                           MLP_conv_stride_nu = MLP_conv_stride_nu,
                           MLP_hidden1_nu = MLP_hidden1_nu, 
                           MLP_hidden2_nu = MLP_hidden2_nu,
                           MLP_lr_nu = MLP_lr_nu,
                           MLP_activation_nu = MLP_activation_nu,
                           MLP_weight_decay_nu = MLP_weight_decay_nu,
                           MLP_epoch_nu = MLP_epoch_nu,
                           MLP_batch_size_nu = MLP_batch_size_nu,
                           train_X=train_x, train_Y=train_y, likelihood=likelihood
                          ).to(torch.float64))
        
        mll = ExactMarginalLogLikelihood(likelihood, gp).to(torch.float64)
        fit_gpytorch_mll_torch(mll)
        ei = UpperConfidenceBound(model=gp, beta = 0.2)
        best_candidate = None
        best_y = float('-inf')
        accuracies = []
        with tqdm(total=n_iter, desc="Bayesian Optimization Progress", unit="iter") as pbar:
            for i in range(n_iter):
                initial_conditions = draw_sobol_samples(bounds=bounds, n=1, q=sample_per_batch).squeeze(1).to(dtype=torch.float64)
                
                candidate, acq_value = self._optimize_acqf_with_discrete_search_space(
                    initial_conditions=initial_conditions,
                    acquisition_function=ei,
                    bounds=bounds,
                    discrete_dims=discrete_dims,
                    discrete_values=discrete_values
                )
                
                train_y = train_y.view(-1, 1)
                new_y = self._botorch_objective(candidate).view(1, 1)
                new_y_value = new_y.item()
                accuracies.append(new_y_value)
                if new_y_value >= best_y:
                    best_y = new_y_value 
                    best_candidate = candidate 
                train_x = torch.cat([train_x, candidate.view(1, -1)])
                train_y = torch.cat([train_y, new_y], dim=0)
                train_y = train_y.view(-1)

                gp.set_train_data(inputs=train_x, targets=train_y, strict=False)
                ei = UpperConfidenceBound(model=gp, beta = 0.2)
                # Update progress bar
                pbar.set_postfix({"Best Y": best_y})  # Dynamically update additional info
                pbar.update(1)  # Increment progress bar by 1

        return accuracies, best_y, best_candidate
    
    def _optimize_acqf_with_discrete_search_space(
        self,
        initial_conditions: torch.Tensor,
        acquisition_function: AcquisitionFunction,
        bounds: torch.Tensor,
        discrete_dims: List[int],
        discrete_values: Dict[int, List[float]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lower_bounds = bounds[0]
        upper_bounds = bounds[1]

        candidates = []
        acq_values = []
        for i in range(initial_conditions.size(0)):
            candidate = initial_conditions[i].clone()

            # Enforce discrete constraints on specified dimensions
            for dim in discrete_dims:
                candidate[dim] = min(discrete_values[dim], key=lambda x: abs(x - candidate[dim]))

            # Enforce bounds by clipping the values
            candidate = torch.max(candidate, lower_bounds)
            candidate = torch.min(candidate, upper_bounds)

            # Evaluate the acquisition function
            acq_value = acquisition_function(candidate.unsqueeze(0))
            candidates.append(candidate)
            acq_values.append(acq_value)

        # Convert lists to tensors
        candidates = torch.stack(candidates)
        acq_values = torch.stack(acq_values)

        return candidates, acq_values


class MLP_GP_model(SingleTaskGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        MLP_conv_feature_num_nu : float,
        MLP_conv_kernel_size_nu : float,
        MLP_conv_stride_nu : float,
        MLP_hidden1_nu: float,
        MLP_hidden2_nu: float,
        MLP_lr_nu: float,
        MLP_activation_nu: float,
        MLP_weight_decay_nu: float,
        MLP_epoch_nu : float,
        MLP_batch_size_nu : float,
        likelihood: Optional[Likelihood],
        train_Yvar: Optional[Tensor] = None,
        outcome_transform: Optional[Union[OutcomeTransform, _DefaultType]] = DEFAULT,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        
        matern_kernel_for_conv_feature_num = ScaleKernel(
            MaternKernel(
                nu = MLP_conv_feature_num_nu,
            )
        )

        matern_kernel_for_conv_kernel_size = ScaleKernel(
            MaternKernel(
                nu = MLP_conv_kernel_size_nu,
            )
        )

        matern_kernel_for_conv_stride = ScaleKernel(
            MaternKernel(
                nu = MLP_conv_stride_nu,
            )
        )


        matern_kernel_for_hidden1 = ScaleKernel(
            MaternKernel(
                nu = MLP_hidden1_nu,
            )
        )

        matern_kernel_for_hidden2 = ScaleKernel(
            MaternKernel(
                nu = MLP_hidden2_nu,
            )
        )

        matern_kernel_for_lr = ScaleKernel(
            MaternKernel(
                nu = MLP_lr_nu,
            )
        )

        matern_kernel_for_activation = ScaleKernel(
            MaternKernel(
                nu = MLP_activation_nu,
            )
        )

        matern_kernel_for_weight_decay = ScaleKernel(
            MaternKernel(
                nu = MLP_weight_decay_nu
            )
        )

        matern_kernel_for_epoch = ScaleKernel(
            MaternKernel(
                nu = MLP_epoch_nu
            )
        )

        matern_kernel_for_batch_size = ScaleKernel(
            MaternKernel(
                nu = MLP_batch_size_nu
            )
        )

        covar_module = (
                        matern_kernel_for_conv_feature_num *
                        matern_kernel_for_conv_kernel_size *
                        matern_kernel_for_conv_stride *
                        matern_kernel_for_hidden1 * 
                        matern_kernel_for_hidden2 * 
                        matern_kernel_for_lr * 
                        matern_kernel_for_activation * 
                        matern_kernel_for_weight_decay * 
                        matern_kernel_for_epoch * 
                        matern_kernel_for_batch_size)

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
        )

