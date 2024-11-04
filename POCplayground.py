from transformers import RobertaTokenizer, RobertaModel

# Load a pre-trained CodeBERT model
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")

code = "def add(a, b): return a + b"
inputs = tokenizer(code, return_tensors="pt")
outputs = model(**inputs)
code_embedding = outputs.last_hidden_state.mean(dim=1)
print(code_embedding)
print(code_embedding.shape)
print(code_embedding.size())