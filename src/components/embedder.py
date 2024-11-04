from src.base.embedder_base import Embedder
from sentence_transformers import SentenceTransformer
import torch

class SentenceEmbedder(Embedder):
    """Vectorizer classs for transforming text to vector rep"""
    def __init__(self, model_type='sentence-transformers/all-MiniLM-L6-v2') -> None:
        super(SentenceEmbedder, self).__init__()
        torch.random.manual_seed(0)
        
        self.model = SentenceTransformer(model_type)


    def __call__(self, input_text: str) -> str:
        """Return vector rep of text"""
        embeddings = self.model.encode(input_text)
        return embeddings
    