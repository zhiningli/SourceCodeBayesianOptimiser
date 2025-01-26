from src.embeddings.model_embedder import Model_Architecture_Code_Embedder, Direct_Model_Code_Embedder
import importlib


direct_embedder = Direct_Model_Code_Embedder()
architecture_embedder = Model_Architecture_Code_Embedder()


for i in range(1, 21):


    module = importlib.import_module(f"src.scripts.models.model{i}")

    model_str = getattr(module, "model", None)

