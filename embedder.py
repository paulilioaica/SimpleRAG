from sentence_transformers import SentenceTransformer
import torch

class Embedder:
    def __init__(self, model_type='sentence-transformers/all-MiniLM-L6-v2') -> None:
        """Vectorizer classs for transforming text to vector rep"""
        torch.random.manual_seed(0)
        
        self.model = SentenceTransformer(model_type)


    def __call__(self, input_text: str) -> str:
        """Return vector rep of text"""
        embeddings = self.model.encode(input_text)
        return embeddings