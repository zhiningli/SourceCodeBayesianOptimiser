from src.data.db.model_crud import ModelRepository
from src.embeddings.source_code_embedders import Codebert_Embedder


repo = ModelRepository()
embedder = Codebert_Embedder()

models = repo.fetch_all_models()

for model_object in models:

    model_name = model_object["model_name"]
    print("model_name: ", model_name)
    model_str = model_object["model_str"]
    updated_field = {
        "model_embeddings": embedder.embed_source_code(code_snippet=model_str)
    }
    repo.update_model(model_name=model_name, updated_fields=updated_field)