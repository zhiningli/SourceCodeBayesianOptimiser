import torch
from torch import Tensor
from botorch.utils.transforms import standardize
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from botorch.models import SingleTaskGP
from botorch.models.model import Model
from botorch.models.gp_regression import SingleTaskGP, ExactGP
from botorch.models.fully_bayesian import MaternKernel
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.utils.transforms import standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.types import _DefaultType, DEFAULT
from typing import Any, Optional, Union, Dict, Tuple, List
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.acquisition import UpperConfidenceBound, AcquisitionFunction, LogExpectedImprovement
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.means.constant_mean import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.utils.grid import scale_to_bounds
from tqdm import tqdm
from botorch.optim.optimize import optimize_acqf_discrete
from itertools import product

class MLP_BO_Optimiser:

    def __init__(self):
        self.params = None
        self.objective_func = None
        self.search_space = {
            'conv_feature_num': [8, 16, 32, 64],
            'conv_kernel_size': [3, 5, 7],
            'conv_stride': [1, 2],
            'hidden1': [64, 128, 256, 512, 1024],
            'lr': [0.0001, 0.0001, 0.001, 0.01],
            'activation': ['ReLU','Tanh','LeakyReLU'],
            'weight_decay': [0.0, 0.1, 0.01, 0.0001, 0.001],
            'epoch': [5, 7, 10, 12, 15, 18, 25, 30, 35],
            'batch_size': [64, 128]
        }
        self.last_error = None

    def optimise(self, code_str, 
                    MLP_conv_feature_num_nu : float,
                    MLP_conv_kernel_size_nu : float,
                    MLP_conv_stride_nu : float,
                    MLP_hidden1_nu: float,
                    MLP_lr_nu: float,
                    MLP_activation_nu: float,
                    MLP_weight_decay_nu: float,
                    MLP_epoch_nu : float,
                    MLP_batch_size_nu : float,
                sample_per_batch=1,
                n_iter=25, 
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
        
        return self._run_bayesian_optimisation(
                                        MLP_conv_feature_num_nu = MLP_conv_feature_num_nu,
                                        MLP_conv_kernel_size_nu = MLP_conv_kernel_size_nu,
                                        MLP_conv_stride_nu = MLP_conv_stride_nu,
                                        MLP_hidden1_nu = MLP_hidden1_nu,
                                        MLP_lr_nu = MLP_lr_nu,
                                        MLP_activation_nu = MLP_activation_nu,
                                        MLP_weight_decay_nu = MLP_weight_decay_nu,
                                        MLP_epoch_nu = MLP_epoch_nu,
                                        MLP_batch_size_nu = MLP_batch_size_nu,
                                        n_iter=n_iter, 
                                        initial_points = initial_points,
                                        sample_per_batch= sample_per_batch)

    def _botorch_objective(self, x):
        """
        A thin wrapper to map input tensor to hyperparameters for MLP
        """
        np_params = x.detach().cpu().numpy().squeeze()
        params = {
            'conv_feature_num': self.search_space['conv_feature_num'][int(np_params[0])],
            'conv_kernel_size': self.search_space['conv_kernel_size'][int(np_params[1])],
            'conv_stride': self.search_space['conv_stride'][int(np_params[2])],
            'hidden1': self.search_space['hidden1'][int(np_params[3])],
            'lr': self.search_space['lr'][int(np_params[4])],
            'activation': self.search_space['activation'][int(np_params[5])],
            'weight_decay': self.search_space['weight_decay'][int(np_params[6])],
            'epoch': self.search_space['epoch'][int(np_params[7])],
            'batch_size': self.search_space['batch_size'][int(np_params[8])],
        }

        return torch.tensor(self.objective_func(**params), dtype=torch.float64)

    def _normalize_to_unit_cube(self, data, bounds):
        lower_bounds = bounds[0].to(data.device)  # Move to the same device as `data`
        upper_bounds = bounds[1].to(data.device)
        return (data - lower_bounds) / (upper_bounds - lower_bounds)


    def _denormalize_from_unit_cube(self, data, bounds):
        lower_bounds = bounds[0].to(data.device)
        upper_bounds = bounds[1].to(data.device)
        return data * (upper_bounds - lower_bounds) + lower_bounds


    def _run_bayesian_optimisation(self, 
                                    n_iter,
                                    initial_points,
                                    sample_per_batch,
                                    MLP_conv_feature_num_nu,
                                    MLP_conv_kernel_size_nu,
                                    MLP_conv_stride_nu,
                                    MLP_hidden1_nu,
                                    MLP_lr_nu,
                                    MLP_activation_nu,
                                    MLP_weight_decay_nu,
                                    MLP_epoch_nu,
                                    MLP_batch_size_nu,
                                   ):
        r"""
        Run Bayesian Optimisation for hyperparameter tuning
        """
        bounds = torch.tensor([
            [0, 0, 0, 0, 0, 0, 0, 0, 0], 
            [4, 3, 2, 5, 4, 3, 5, 9, 2]   
        ], dtype=torch.float64)

        if torch.cuda.is_available():
            bounds = bounds.cuda()  # Move bounds to GPU

        choices = torch.tensor(list(product(*[range(len(self.search_space[dim])) for dim in self.search_space])), dtype=torch.float64)
        print(choices[:5])

        print("Running bayesian optimisation...")

        train_x = draw_sobol_samples(bounds=bounds, n=initial_points, q=sample_per_batch).squeeze(1)
        print("initial point", train_x)
        train_y = torch.tensor([self._botorch_objective(x).item() for x in train_x], dtype=torch.float64).view(-1, 1)

        train_x = self._normalize_to_unit_cube(train_x, bounds)

        if torch.cuda.is_available():
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            choices = choices.cuda()

        likelihood = GaussianLikelihood().to(torch.float64)
        gp = (MLP_GP_model(
            MLP_conv_feature_num_nu = MLP_conv_feature_num_nu,
            MLP_conv_kernel_size_nu = MLP_conv_kernel_size_nu,
            MLP_conv_stride_nu = MLP_conv_stride_nu,
            MLP_hidden1_nu = MLP_hidden1_nu,
            MLP_lr_nu = MLP_lr_nu,
            MLP_activation_nu = MLP_activation_nu,
            MLP_weight_decay_nu = MLP_weight_decay_nu,
            MLP_epoch_nu = MLP_epoch_nu,
            MLP_batch_size_nu = MLP_batch_size_nu,
            train_X = train_x,
            train_Y= train_y,
            likelihood=likelihood,
        ).to(torch.float64))

        if torch.cuda.is_available():

            likelihood = likelihood.cuda()
            gp = gp.cuda()

        mll = ExactMarginalLogLikelihood(likelihood, gp).to(torch.float64)
        fit_gpytorch_mll_torch(mll)
        acq_function = UpperConfidenceBound(model = gp, beta = 0.01)

        best_candidate = None
        best_y = float('-inf')
        accuracies = []
        with tqdm(total=n_iter, desc="Bayesian Optimization Progress", unit="iter") as pbar:
            for i in range(n_iter):
                candidate, acq_value = optimize_acqf_discrete(
                    acq_function=acq_function,
                    q=1,
                    choices=choices,  # Raw choices, not normalized
                    max_batch_size=2048,
                    unique=True
                )

                print(f"Raw Candidate: {candidate} with acquisition value {acq_value}")

                # Normalize candidate
                candidate = self._normalize_to_unit_cube(candidate, bounds)

                # Ensure candidate is on the same device as train_x
                if torch.cuda.is_available():
                    candidate = candidate.cuda()

                # Evaluate the objective function
                new_y = self._botorch_objective(candidate).view(1, 1)
                new_y = new_y.to(train_y.device)
                new_y_value = new_y.item()

                if new_y_value >= best_y:
                    best_y = new_y_value
                    best_candidate = self._denormalize_from_unit_cube(candidate, bounds)

                accuracies.append(new_y_value)

                # Update train_x and train_y
                train_x = torch.cat([train_x, candidate.view(1, -1)])
                train_y = train_y.view(-1, 1)
                train_y = torch.cat([train_y, new_y], dim=0)
                train_y = train_y.view(-1)

                # Update the GP model
                gp.set_train_data(inputs=train_x, targets=train_y, strict=False)
                acq_function = UpperConfidenceBound(model = gp, beta = 0.01)

                # Update progress bar
                pbar.set_postfix({"Best Y": best_y})
                pbar.update(1)


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


class MLP_GP_model_addition_kernel(SingleTaskGP):
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
                        matern_kernel_for_conv_feature_num +
                        matern_kernel_for_conv_kernel_size +
                        matern_kernel_for_conv_stride +
                        matern_kernel_for_hidden1 +
                        matern_kernel_for_hidden2 + 
                        matern_kernel_for_lr + 
                        matern_kernel_for_activation + 
                        matern_kernel_for_weight_decay + 
                        matern_kernel_for_epoch + 
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


class DKL(torch.nn.Sequential):

    def __init__(self):
        super(DKL, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(9, 9),
            torch.nn.ReLU(),
            torch.nn.Linear(9, 9)
        )


class MLP_GP_DKL(SingleTaskGP):

    def __init__(self, train_x, train_y, likelihood,
                train_Yvar: Optional[Tensor] = None,
                outcome_transform: Optional[Union[OutcomeTransform, _DefaultType]] = DEFAULT,
                input_transform: Optional[InputTransform] = None,):
        super().__init__(train_X = train_x, train_Y = train_y, likelihood = likelihood, train_Yvar= train_Yvar, outcome_transform= outcome_transform, input_transform= input_transform)
        self.DKL = DKL()
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
    
    def forward(self, x: Tensor) -> MultivariateNormal:

        transformed_x = scale_to_bounds(self.DKL(x), -1., 1.)
        mean_x = self.mean_module(transformed_x)
        covar_x = self.covar_module(transformed_x)
        print("Mean:", mean_x)
        print("Covariance Matrix Norm:", covar_x.evaluate().norm().item())
        print("Lengthscale:", self.covar_module.base_kernel.lengthscale)
        print("Outputscale:", self.covar_module.outputscale)
        return MultivariateNormal(mean_x, covar_x)