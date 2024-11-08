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
    

class TrainsformedOverlapKernel(Kernel):
    def __init__(self, num_dimensions, **kwargs):
        super(TrainsformedOverlapKernel, self).__init__(has_lengthscale=True, **kwargs)

        self.register_parameter(
            name = "raw_lengthscale",
            parameter=torch.nn.Parameter(torch.ones(1, num_dimensions))
        ) 

        self.register_constraint("raw_lengthscale", Positive())
    
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if diag:
            matches = (x1 == x2).float()
            weighted_sum = torch.sum(self.lengthscale * matches, dim=-1)
            return torch.exp(weighted_sum)
    
        x1_expanded = x1.unsqueeze(1)
        x2_expanded = x2.unsqueeze(0)

        matches = (x1_expanded == x2_expanded).float()
        weighted_sum = torch.sum(self.lengthscale * matches, dim=-1)

        return torch.exp(weighted_sum)
    
    