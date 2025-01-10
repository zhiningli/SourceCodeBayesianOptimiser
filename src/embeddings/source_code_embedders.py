from transformers import RobertaTokenizer, RobertaModel
import torch


class Codebert_Embedder:

    def __init__(self):
        self.tokeniser = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")

    def embed_source_code(self, code_snippet):

        inputs = self.tokeniser(code_snippet, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings