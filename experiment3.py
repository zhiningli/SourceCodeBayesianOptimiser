from src.embeddings.model_embedder import Model_Architecture_Code_Embedder, Direct_Model_Code_Embedder
import importlib
from src.similarities import CosineSimilarity, EuclideanSimilarity

# Initialize embedders
direct_embedder = Direct_Model_Code_Embedder()
architecture_embedder = Model_Architecture_Code_Embedder()

# Initialize similarity calculator
similarity_computer = CosineSimilarity()
euclidean_similarity_computer = EuclideanSimilarity()

# Initialize dictionaries for models and corrupted models
models = {}
corrupted_models = {}

for i in range(11, 21):
    module = importlib.import_module(f"src.scripts.models.model{i}")

    try:
        model_str = getattr(module, "model")
        corrupted_model_str = getattr(module, "corrupted_model")
    except AttributeError as e:
        print(f"Model {i}: Error loading model strings: {e}")
        continue

    if model_str is None or corrupted_model_str is None:
        print(f"Model {i}: Missing model or corrupted model string.")
        continue

    models[i] = {}
    corrupted_models[i] = {}

    models[i]["direct_embedding"] = direct_embedder.embed_source_code(model_str)
    models[i]["architecture_embedding"] = architecture_embedder.embed_source_code(model_str)

    corrupted_models[i]["direct_embedding"] = direct_embedder.embed_source_code(corrupted_model_str)
    corrupted_models[i]["architecture_embedding"] = architecture_embedder.embed_source_code(corrupted_model_str)

# Experiment part A
print("Experiment A: Running models direct embedding against a list of corrupted models' embeddings, hoping to identify the model that is functionally identical")
for i in range(11, 21):
    if i not in models:
        continue

    current_direct_embedding = models[i]["direct_embedding"]
    res = []

    for j, corrupted_embedding_object in corrupted_models.items():
        corrupted_embedding = corrupted_embedding_object["direct_embedding"]
        similarity_score = similarity_computer.compute(current_direct_embedding, corrupted_embedding).item()
        res.append((j, similarity_score))

    res.sort(key=lambda x: -x[1])
    print(f"Model {i}: Most similar corrupted model by direct embedding: {res}")

print("\n" * 5)
print("Experiment B")
print("Experiment B: Running models' architecture embedding against a list of corrupted models' embeddings, hoping to identify the model that is functionally identical")
# Experiment part B
for i in range(11, 21):
    if i not in models:
        continue

    current_architecture_embedding = models[i]["architecture_embedding"]
    res = []

    for j, corrupted_embedding_object in corrupted_models.items():
        corrupted_embedding = corrupted_embedding_object["architecture_embedding"]
        similarity_score = similarity_computer.compute(current_architecture_embedding, corrupted_embedding).item()
        res.append((j, similarity_score))

    res.sort(key=lambda x: -x[1])
    print(f"Model {i}: Most similar corrupted model by architecture embedding: {res}")

print("\n" * 5)
print("Experiment C: Running corrupted models' direct embedding against a list of models' embeddings, hoping to identify the model that is functionally identical")
# Experiment part C
for i in range(11, 21):
    if i not in models:
        continue

    current_direct_embedding = corrupted_models[i]["direct_embedding"]
    res = []

    for j, embedding_object in models.items():
        model_embedding = embedding_object["direct_embedding"]
        similarity_score = similarity_computer.compute(current_direct_embedding, model_embedding).item()
        res.append((j, similarity_score))

    res.sort(key=lambda x: -x[1])
    print(f"Corrupted Model {i}: Most similar model by architecture embedding: {res}")

# Experiment part D
print("Experiment D: Running corrupted models' architecture embedding against a list of models' embeddings, hoping to identify the model that is functionally identical")
print("\n" * 5)
for i in range(11, 21):
    if i not in models:
        continue

    current_architecture_embedding = corrupted_models[i]["architecture_embedding"]
    res = []

    for j, embedding_object in models.items():
        model_embedding = embedding_object["architecture_embedding"]
        similarity_score = similarity_computer.compute(current_architecture_embedding, model_embedding).item()
        res.append((j, similarity_score))

    res.sort(key=lambda x: -x[1])
    print(f"Corrupted Model {i}: Most similar model by architecture embedding: {res}")


print("\n" * 5)
# Experiment part E
print("Experiment E: Running models' architecture embedding against a list of models' embeddings, hoping to identify itself")
for i in range(11, 21):
    if i not in models:
        continue

    current_architecture_embedding = models[i]["architecture_embedding"]
    res = []

    for j, embedding_object in models.items():
        model_embedding = embedding_object["architecture_embedding"]
        similarity_score = similarity_computer.compute(current_architecture_embedding, model_embedding).item()
        res.append((j, similarity_score))

    res.sort(key=lambda x: -x[1])
    print(f"Model {i}: Most similar model by architecture embedding: {res}")

print("\n" * 5)
# Experiment part F
print("Experiment F: Similar to E but uses Euclidean Similarity instead of Cosine Similarity")
for i in range(11, 21):
    if i not in models:
        continue

    current_architecture_embedding = models[i]["architecture_embedding"]
    res = []

    for j, embedding_object in models.items():
        model_embedding = embedding_object["architecture_embedding"]
        similarity_score = euclidean_similarity_computer.compute(current_architecture_embedding, model_embedding).item()
        res.append((j, similarity_score))

    res.sort(key=lambda x: -x[1])
    print(f"Model {i}: Most similar model by architecture embedding: {res}")

print("\n" * 5)
# Experiment part G
print("Experiment G: Similar to A but uses Euclidean Similarity instead of Cosine Similarity")
for i in range(11, 21):
    if i not in models:
        continue

    current_direct_embedding = models[i]["direct_embedding"]
    res = []

    for j, corrupted_embedding_object in corrupted_models.items():
        corrupted_embedding = corrupted_embedding_object["direct_embedding"]
        similarity_score = euclidean_similarity_computer.compute(current_direct_embedding, corrupted_embedding).item()
        res.append((j, similarity_score))

    res.sort(key=lambda x: -x[1])
    print(f"Model {i}: Most similar corrupted model by direct embedding: {res}")


print("\n" * 5)
# Experiment part H
print("Experiment H: Similar to G but uses Euclidean Similarity instead of Cosine Similarity")
for i in range(11, 21):
    if i not in models:
        continue

    current_direct_embedding = models[i]["architecture_embedding"]
    res = []

    for j, corrupted_embedding_object in corrupted_models.items():
        corrupted_embedding = corrupted_embedding_object["architecture_embedding"]
        similarity_score = euclidean_similarity_computer.compute(current_direct_embedding, corrupted_embedding).item()
        res.append((j, similarity_score))

    res.sort(key=lambda x: -x[1])
    print(f"Model {i}: Most similar corrupted model by architecture embedding: {res}")