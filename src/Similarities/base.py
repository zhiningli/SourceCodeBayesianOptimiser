import torch
import torch.nn.functional as F

class SimilarityBase:

    def compute(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> torch.Tensor:
        """
        Abstract method to compute similarity between two embeddings
        """
        if not (isinstance(embedding1, torch.Tensor) and (isinstance(embedding2, torch.Tensor))):
            raise TypeError(f"Expected both embeddings to be Pytorch tensor, got first embedding: {type(embedding1)}, second embedding: {type(embedding2)}")

        if embedding1.shape != embedding2.shape:
            raise ValueError(f"Incompatible shape for two embeddings, embedding1 has shape of {embedding1.shape}, embedding2 has shape of{embedding2.shape} ")

        raise NotImplementedError("This method must be inherited and implemented by children classes")


class ConsineSimilarity(SimilarityBase):

    def compute(self, embedding1: torch.Torch , embedding2: torch.Torch) -> torch.Torch:
        r"""
        Compute a cosine similarity between two PyTorch Tensors (or NumPy arrays).
        
        Params:
        embedding1: torch.Tensor or np.ndarray
            First embedding vector.
        embedding2: torch.Tensor or np.ndarray
            Second embedding vector.
        
        Output:
            torch.Tensor showing the cosine similarity.
        """
        return F.cosine_similarity(embedding1, embedding2, dim=0)


    