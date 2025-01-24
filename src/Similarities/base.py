import numpy as np
import torch

from typing import Union

class SimilarityBase:

    def __init__(self):
        self.inputType = None

    def compute(self, embedding1: Union[np.ndarray, torch.Tensor], embedding2: Union[np.ndarray, torch.Tensor]):
        """
        Abstract method to compute similarity between two embeddings
        """

    def _prepare_embeddings(self, embedding):
        """
        Typing and shape checking for embeddings
        """
        if isinstance(embedding, torch.Tensor):
            pass
    
    def _transform_to_numpy_array(self, embedding_tensor: torch.Tensor) -> np.ndarray:
        """
        Transform a Pytorch tensor into a Numpy array, handling edge cases
        """