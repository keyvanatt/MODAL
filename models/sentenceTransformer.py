import torch
from sentence_transformers import SentenceTransformer

import torch.nn as nn

class SentenceTransformerEncoder(nn.Module):
    def __init__(self, model_name='all-MiniLM-L6-v2', device=None):
        super().__init__()
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def forward(self, texts):
        """
        Args:
            texts (list of str): List of sentences to encode.
        Returns:
            torch.Tensor: Encoded sentence embeddings (batch_size, embedding_dim)
        """
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True,show_progress_bar=False)
        return embeddings

# Example usage:
# encoder = SentenceTransformerEncoder()
# embeddings = encoder(["Hello world!", "How are you?"])