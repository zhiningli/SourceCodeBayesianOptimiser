import torch
from typing import List, Dict, Tuple
from botorch.acquisition.analytic import UpperConfidenceBound

# Define the mock function and parameters
class MockAcquisitionFunction:
    def __call__(self, x):
        # Return a simple function value for testing (e.g., sum of the tensor)
        return torch.sum(x, dim=-1)

# Define a small test case
bounds = torch.tensor([[0, 0, 0], [3, 4, 2]], dtype=torch.float32)  # 3D bounds
discrete_dims = [0, 1, 2]
discrete_values = {
    0: [0, 1, 2, 3],    # Possible values for dimension 0
    1: [0, 2, 4],        # Possible values for dimension 1
    2: [0, 1, 2]         # Possible values for dimension 2
}

initial_conditions = torch.tensor([[1.5, 3.7, 1.1]], dtype=torch.float32)  # One starting point
acquisition_function = MockAcquisitionFunction()

# Define the test function
def test_optimize_acqf_with_discrete_search_space():
    def _optimize_acqf_with_discrete_search_space(
        initial_conditions: torch.Tensor,
        acquisition_function: MockAcquisitionFunction,
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

    # Run the function
    candidate, acq_value = _optimize_acqf_with_discrete_search_space(
        initial_conditions=initial_conditions,
        acquisition_function=acquisition_function,
        bounds=bounds,
        discrete_dims=discrete_dims,
        discrete_values=discrete_values
    )

    # Print results for validation
    print("Candidate:", candidate)
    print("Acquisition Value:", acq_value)

# Execute the test
test_optimize_acqf_with_discrete_search_space()
