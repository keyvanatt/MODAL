from transformers import DistilBertTokenizer, DistilBertModel
import torch
import torch.nn as nn

class DistilBert(nn.Module):
    def __init__(self, text_model_name, freeze=True, freeze_last_layers=False):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained(text_model_name)
        self.text_encoder = DistilBertModel.from_pretrained(text_model_name)
        if freeze:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            if not freeze_last_layers:
                for name, param in list(self.text_encoder.named_parameters())[-2:]:
                    param.requires_grad = False
        self.dim = self.text_encoder.config.hidden_size

    def forward(self, input_text,device):
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=30)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)
        title_feat = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return title_feat