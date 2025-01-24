import torch
import numpy as np

def torch_tensor_to_numpy_array(torch_tensor: torch.Tensor) -> np.ndarray:
    """
    Transform a Pytorch tensor into a Numpy array
    """
    if not isinstance(torch_tensor, torch.Tensor):
        raise TypeError(f"Expect a torch.Tensor object as input, got {type(torch_tensor)}")
    
    

    return torch_tensor.detach().numpy()
