import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

class DistilBertEncoder(nn.Module):
    def __init__(self, model_name="distilbert/distilbert-base-multilingual-cased", pool=True, freeze=True):
        super(DistilBertEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        if not freeze:
            for param in list(self.model.parameters())[-2:]:
                param.requires_grad = False
        self.dim = self.model.config.hidden_size
        self.pool = pool

    def forward(self, texts):
        """
        Encode a batch of texts.
        
        Args:
            texts (List[str]): A list of input strings.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, hidden_size) with text embeddings.
        """
        # Tokenize input texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        # Move inputs to the same device as the model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state

        if self.pool:
            # Pooling: mean over sequence length
            pooled = last_hidden_state.mean(dim=1)
            return pooled
        else:
            # Return the last hidden state
            return last_hidden_state

# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertEncoder(pool=False).to(device)
    
    texts = ["Bonjour", "Hello", "Hola"]
    embeddings = model(texts)
    print(embeddings.shape) 
