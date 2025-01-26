import torch
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models import SingleTaskGP
from botorch.models.gp_regression import SingleTaskGP
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.fit import fit_gpytorch_mll_torch
from botorch.acquisition import UpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from tqdm import tqdm
from botorch.optim.optimize import optimize_acqf_discrete
from itertools import product
from src.middleware import ComponentStore
from typing import Callable, Dict, List, Tuple

class MLP_BO_Optimiser:

    def __init__(self):
        self.objective_func: Callable = None
        self.search_space: Dict[str, List[float]] = None
        self.bounds: torch.Tensor = None
        self._store: ComponentStore = None

    @property
    def store(self) -> ComponentStore:
        return self._store
    
    @store.setter
    def store(self, value: ComponentStore) -> None:
        self._store = value

    def optimise(self,
                    search_space: Dict[str, List[float]],
                    sample_per_batch: int = 1,
                    n_iter: int = 20, 
                    initial_points: int = 25,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] :
        r"""
        Optimize the hyperparameters using Bayesian Optimization.

        Params:
        search_space: A dict object with string being the search space name and value being the search space range
        sample_per_patch: int specifying sample per batch
        n_iter: iteration count
        initial_points: number of initial point to sample
        

        Returns:
        
        Tuple containing:
        Accuracies: torch.Torch recording the accuracies along the way
        Best_y: torch scalar Tensor showing the best way
        Best_candidate: torch.Tensor showing the minimum found
        """
        self.search_space = search_space
        self.objective_func = self.store.objective_func
        self.bounds = torch.Tensor([
                                    [0, 0, 0, 0],
                                    [len(self.search_space['learning_rate'])-1, 
                                     len(self.search_space['momentum'])-1, 
                                     len(self.search_space['weight_decay'])-1, 
                                     len(self.search_space['num_epochs'])-1]
                                    ], )

        if not self.objective_func:
            raise ValueError("Objective function not loaded to the bayesian optimiser, check if Component Store is initiated")

        if not callable(self.objective_func):
            raise ValueError("Unable to execute the objective function")
        
        return self._run_bayesian_optimisation(
                                        n_iter=n_iter, 
                                        initial_points = initial_points,
                                        sample_per_batch= sample_per_batch)

    def _botorch_objective(self, x: torch.Tensor) -> torch.Tensor:
        """
        A thin wrapper to map input tensor to hyperparameters for MLP
        """
        np_params = x.detach().cpu().numpy().squeeze()
        print("current Index: ", np_params)
        params = {
            "learning_rate": self.search_space["learning_rate"][int(np_params[0])],
            "momentum": self.search_space["momentum"][int(np_params[1])],
            "weight_decay": self.search_space["weight_decay"][int(np_params[2])],
            "num_epochs": self.search_space["num_epochs"][int(np_params[3])]
        }

        print("current X: ", params)

        return torch.tensor(self.objective_func(**params), dtype=torch.float64)

    def _normalize_to_unit_cube(self, data: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        lower_bounds = bounds[0].to(data.device) 
        upper_bounds = bounds[1].to(data.device)
        return (data - lower_bounds) / (upper_bounds - lower_bounds)


    def _denormalize_from_unit_cube(self, data: torch.Tensor, bounds: torch.Tensor) -> torch.Tensor:
        lower_bounds = bounds[0].to(data.device)
        upper_bounds = bounds[1].to(data.device)
        return data * (upper_bounds - lower_bounds) + lower_bounds


    def _run_bayesian_optimisation(self, 
                                    n_iter: int,
                                    initial_points: int,
                                    sample_per_batch: int,
                                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Run Bayesian Optimisation for hyperparameter tuning
        """
        bounds = self.bounds
        if torch.cuda.is_available():
            bounds = self.bounds.cuda()

        train_x = draw_sobol_samples(bounds=bounds, n=initial_points, q=sample_per_batch).squeeze(1).cuda()
        train_x = train_x.to(torch.float64)
        train_y = torch.tensor([self._botorch_objective(x).item() for x in train_x], dtype=torch.float64).view(-1, 1)

        normalised_train_x = self._normalize_to_unit_cube(train_x, bounds)

        choices = torch.tensor(list(product(*[range(len(self.search_space[dim])) for dim in self.search_space])), dtype=torch.float64)

        normalized_choices = choices / torch.tensor(
            [len(self.search_space[dim]) - 1 for dim in self.search_space],
            dtype=torch.float64
        )

        if torch.cuda.is_available():
            normalised_train_x = normalised_train_x.cuda()
            train_y = train_y.cuda()
            normalized_choices = normalized_choices.cuda()

        likelihood = GaussianLikelihood().to(torch.float64)
        gp = (SingleTaskGP(
            train_X = normalised_train_x,
            train_Y= train_y,
            likelihood=likelihood,
        ).to(torch.float64))

        if torch.cuda.is_available():
            likelihood = likelihood.cuda()
            gp = gp.cuda()

        mll = ExactMarginalLogLikelihood(likelihood, gp).to(torch.float64)
        fit_gpytorch_mll_torch(mll)
        acq_function = UpperConfidenceBound(model = gp, beta = 2)

        best_candidate = None
        best_y = float('-inf')
        accuracies = []
        with tqdm(total=n_iter, desc="Bayesian Optimization Progress", unit="iter") as pbar:
            for i in range(n_iter):
                candidate, acq_value = optimize_acqf_discrete(
                    acq_function=acq_function,
                    q=1,
                    choices=normalized_choices, 
                    max_batch_size=2048,
                    unique=True
                )
                
                # Normalize candidate to evaluate y
                candidate = self._denormalize_from_unit_cube(candidate, bounds)
                if torch.cuda.is_available():
                    candidate = candidate.cuda()
                
                new_y = self._botorch_objective(candidate).view(1, 1)
                new_y = new_y.to(train_y.device)
                new_y_value = new_y.item()

                if new_y_value >= best_y:
                    best_y = new_y_value
                    best_candidate = candidate

                candidate = self._normalize_to_unit_cube(candidate, bounds)

                accuracies.append(new_y_value)

                normalised_train_x = torch.cat([normalised_train_x, candidate.view(1, -1)])
                
                train_y = train_y.view(-1, 1)
                train_y = torch.cat([train_y, new_y], dim=0)
                train_y = train_y.view(-1)

                gp.set_train_data(inputs=normalised_train_x, targets=train_y, strict=False)
                acq_function = UpperConfidenceBound(model = gp, beta = 2)

                pbar.set_postfix({"Best Y": best_y})
                pbar.update(1)
        return accuracies, best_y, best_candidate