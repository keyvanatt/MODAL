from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn

class LlamaTextEncoder(nn.Module):
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", pool: bool = False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_size = self.model.config.hidden_size
        self.dim = self.hidden_size
        self.pool = pool

    def forward(self, text, attention_mask=None):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        # Use the last hidden state (batch, seq_len, hidden_size)
        last_hidden_state = outputs.last_hidden_state
        # Pooling: mean over sequence length
        if self.pool:
            pooled = last_hidden_state.mean(dim=1)
            return pooled # (batch, hidden_size = dim)
        else:
            return last_hidden_state
        