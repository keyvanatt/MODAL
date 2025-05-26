from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch
import os
from transformers import BitsAndBytesConfig
torch.cuda.empty_cache()


class LlamaTextEncoder(nn.Module):
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf", pool: bool = False, freeze: bool = True, device = "cuda"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Use EOS as PAD if no pad token exists
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)

        for param in self.model.parameters():
            param.requires_grad = False
        self.hidden_size = self.model.config.hidden_size
        self.dim = 2048
        self.pool = pool
        self.device = device
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.dim),
            nn.Dropout(0.2),
        )


    def forward(self, text, attention_mask=None):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True,max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move inputs to the specified device
        outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)
        last_hidden_state = self.projector(last_hidden_state.to(torch.float32))  # Project to desired dimension

        # Pooling: mean over sequence length
        if self.pool:
            pooled = last_hidden_state.mean(dim=1)
            return pooled # (batch, hidden_size = dim)
        else:
            return last_hidden_state


if __name__ == "__main__":
    model = LlamaTextEncoder(pool=False)
    text = ["Hello, world!", "This is a big code i wanna do, bc i like coding."]
    model.to("cuda")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        print(torch.cuda.memory_summary())
    else:
        print("CUDA is not available.")
    outputs = model(text)
    print(outputs) 
    print(outputs.shape)  # Should print (batch_size, seq_len, hidden_size)
    print(model.dim)  # Should print the hidden size
        