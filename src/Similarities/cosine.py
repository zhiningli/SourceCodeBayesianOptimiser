from .base import SimilarityBase

class ConsineSimilarity(SimilarityBase):

    def compute(self, embedding1, embedding2):
        super().compute(embedding1 = embedding1, embedding2 = embedding2)
        
