from src.embeddings.source_code_embedders import Codebert_Embedder
from src.scripts.full_script.scripts12 import code_str
from src.scripts.models.model1 import model as model1
from src.scripts.models.model5 import model as model5
import torch.nn.functional as F

model_code_str1 = model1
model_code_str2 = model5

embedder = Codebert_Embedder()

embedding1 = embedder.embed_source_code(model_code_str1)
embedding2 = embedder.embed_source_code(model_code_str2)

print(type(embedding1))
similarity = F.cosine_similarity(embedding1, embedding2)

print("Similarity between this two models: ", similarity)
