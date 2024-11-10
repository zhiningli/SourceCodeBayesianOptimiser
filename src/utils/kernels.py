import torch
from torch import nn, Tensor
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive
from typing import Optional, Union

class TransformedOverlapKernel(Kernel):
    def __init__(
        self,
        lengthscale_prior: Optional[nn.Module] = None,
        lengthscale_constraint: Optional[Positive] = None,
        sigma: float = 1.0,  # Scaling factor (σ)
        normalization_constant: float = 1.0,  # Normalization constant (c)
        **kwargs
    ):
        super(TransformedOverlapKernel, self).__init__(has_lengthscale=True, **kwargs)

        lengthscale_constraint = Positive() if lengthscale_constraint is None else lengthscale_constraint
        self.register_parameter(
            name="raw_lengthscale",
            parameter=nn.Parameter(torch.ones(*self.batch_shape, 1))
        )
        if lengthscale_prior is not None:
            self.register_prior(
                "lengthscale_prior",
                lengthscale_prior,
                lambda m: m.lengthscale,
                lambda m, v: m._set_lengthscale(v)
            )
        self.register_constraint("raw_lengthscale", lengthscale_constraint)

        self.sigma = sigma
        self.normalization_constant = normalization_constant

    @property
    def lengthscale(self) -> Tensor:
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    @lengthscale.setter
    def lengthscale(self, value: Union[float, Tensor]):
        self._set_lengthscale(value)

    def _set_lengthscale(self, value: Union[float, Tensor]):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_lengthscale)
        self.initialize(raw_lengthscale=self.raw_lengthscale_constraint.inverse_transform(value))

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: Optional[bool] = False,
        last_dim_is_batch: Optional[bool] = False,
        **params
    ) -> Tensor:
        # Ensure inputs have consistent dimensions for broadcasting
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        # Compute element-wise indicator function δ(h_i - h'_i)
        x1_expanded = x1.unsqueeze(1) if x1.dim() == 2 else x1
        x2_expanded = x2.unsqueeze(0) if x2.dim() == 2 else x2

        # Check for diagonal computation
        if diag:
            if torch.equal(x1, x2):
                weighted_sum = torch.sum(self.lengthscale, dim=-1)
                kernel_matrix = torch.exp((self.sigma ** 2 / self.normalization_constant) * weighted_sum)
                return kernel_matrix.expand(x1.size(0))
            else:
                matches = (x1 * x2).sum(dim=-1)
                dist = x1.size(-1) - matches
                weighted_sum = self.lengthscale * dist
                kernel_matrix = torch.exp((self.sigma ** 2 / self.normalization_constant) * weighted_sum)
                return kernel_matrix.diagonal(dim1=-2, dim2=-1)

        # Compute matches for non-diagonal case
        matches = (x1_expanded == x2_expanded).float()  # Shape: (n1, n2, d)
        weighted_sum = torch.sum(self.lengthscale * matches, dim=-1)  # Shape: (n1, n2)

        # Apply exponential transformation
        kernel_matrix = torch.exp((self.sigma ** 2 / self.normalization_constant) * weighted_sum)

        return kernel_matrix
