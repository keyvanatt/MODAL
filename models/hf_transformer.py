# hf_transformer.py

import torch.nn as nn
from transformers import AutoModel

class HFTransformer(nn.Module):
    def __init__(self, pretrained_model_name_or_path: str, num_classes: int, hidden_dropout_prob: float = 0.1, attention_probs_dropout_prob: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            pretrained_model_name_or_path,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, batch):
        outputs = self.backbone(**{"pixel_values": batch["image"]})
        pooled = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled)
