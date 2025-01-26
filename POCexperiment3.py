from src.scripts.full_script.scripts95 import code_str
from src.middleware import ComponentStore
from src.embeddings.model_embedder import Model_Architecture_Code_Embedder

store = ComponentStore()
store.code_string = code_str
embedder = Model_Architecture_Code_Embedder()

embedder.embed_source_code(store.code_string)

