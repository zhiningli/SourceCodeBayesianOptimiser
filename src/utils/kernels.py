import torch
import gpytorch
from gpytorch.kernels import Kernel
from gpytorch.constraints import Positive

class HammingKernel(Kernel):

    def __init__(self, **kwargs):
        super(HammingKernel, self).__init__(has_lengthscale=False, **kwargs)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if diag:
            return torch.sum((x1==x2).float(), dim=-1) / x1.size(-1)
        
        x1_expanded = x1.unsqueeze(1)
        x2_expanded = x2.unsqueeze(0)

        matches = (x1_expanded == x2_expanded).float()
        hamming_similarity = torch.mean(matches, dim=-1)

        return hamming_similarity


class TransformedOverlapKernel(Kernel):
    def __init__(self, num_dimensions, **kwargs):
        super(TransformedOverlapKernel, self).__init__(has_lengthscale=True, **kwargs)

        self.register_parameter(
            name="raw_lengthscale",
            parameter=torch.nn.Parameter(torch.ones(1, num_dimensions))
        )
        
        self.register_constraint("raw_lengthscale", Positive())

    @property
    def lengthscale(self):
        return self.raw_lengthscale_constraint.transform(self.raw_lengthscale)

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # Check if diagonal computation is requested
        if diag:
            matches = (x1 == x2).float()
            weighted_sum = torch.sum(self.lengthscale * matches, dim=-1)
            return torch.exp(weighted_sum)
        
        # Expand x1 and x2 to align for broadcasting
        # Corrected shape handling
        x1_expanded = x1.unsqueeze(1)  # Shape: (n1, 1, d)
        x2_expanded = x2.unsqueeze(0)  # Shape: (1, n2, d)

        # Compute element-wise matches and apply lengthscale weighting
        matches = (x1_expanded == x2_expanded).float()  # Shape: (n1, n2, d)
        weighted_sum = torch.sum(self.lengthscale * matches, dim=-1)  # Shape: (n1, n2)

        # Check output shape
        print("Corrected shape of weighted_sum:", weighted_sum.shape)

        return torch.exp(weighted_sum)



    
