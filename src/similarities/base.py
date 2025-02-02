import torch
import torch.nn.functional as F
from abc import ABC, abstractclassmethod

class SimilarityBase(ABC):

    @abstractclassmethod
    def compute(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to compute similarity between two embeddings
        """
        if not (isinstance(embedding1, torch.Tensor) and (isinstance(embedding2, torch.Tensor))):
            raise TypeError(f"Expected both embeddings to be Pytorch tensor, got first embedding: {type(embedding1)}, second embedding: {type(embedding2)}")

        if embedding1.shape != embedding2.shape:
            raise ValueError(f"Incompatible shape for two embeddings, embedding1 has shape of {embedding1.shape}, embedding2 has shape of{embedding2.shape} ")

        raise NotImplementedError("This method must be inherited and implemented by children classes")

    def to_similarity(self, distance: torch.Tensor):
        r"""
        Convert distance into a similarity metric between 0 and 1 via reciproval transformation
        
        Params:
        distance: torch.Tensor
        
        Returns:
        similarity: torch.Tensor
        A scalar tensor as similarity score
        """
        if not isinstance(distance, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor, but got {type(distance)}")
        
        if distance.dim() != 0:
            raise ValueError(f"Expected a scalar tensor (dim=0), but got a tensor with shape {distance.shape}")
        
        return 1. / (1. + distance)


class CosineSimilarity(SimilarityBase):

    def compute(self, embedding1: torch.Tensor , embedding2: torch.Tensor) -> torch.Tensor:
        r"""
        Compute a cosine similarity between two PyTorch Tensors (or NumPy arrays).
        
        Params:
        embedding1: torch.Tensor or np.ndarray
            First embedding vector.
        embedding2: torch.Tensor or np.ndarray
            Second embedding vector.
        
        Returns:
            torch.Tensor 
            A scalar tensor representing the cosine similarity between 0 and 1.
        """
        embedding1 = embedding1.squeeze(0) 
        embedding2 = embedding2.squeeze(0) 
        return F.cosine_similarity(embedding1, embedding2, dim=0)


class EuclideanSimilarity(SimilarityBase):
    def compute(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the Enclidean similarity between two embeddings in the form of Pytorch Tensors

        Params:
        embedding1: torch.Tensor
            First embedding tensor
        embedding2: torch.Tensor
            Second embedding tensor
        
        Returns:
            torch.Tensor:
                A scalar tensor representing the Euclidean distance    
        """
        # Squeeze the tensors to remove batch dimension (if present)
        embedding1 = embedding1.squeeze(0)  # Shape becomes (D,)
        embedding2 = embedding2.squeeze(0)  # Shape becomes (D,)

        # Compute the Euclidean distance
        distance = torch.dist(embedding1, embedding2, p=2)

        # Convert distance to similarity (optional, based on your use case)
        return self.to_similarity(distance)


class ManhanttanSimilarity(SimilarityBase):
    def compute(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the Manhanttan distance between two embeddings in the form of Pytorch Tensors

        Params:
        embedding1: torch.Tensor
            First embedding tensor
        embedding2: torch.Tensor
            Second embedding tensor
        
        Returns:
            torch.Tensor:
                A scalar tensor representing the Manhanttan distance    
        """

        return self.to_similarity(torch.dist(embedding1, embedding2, p = 1))
    
