import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from dinov2 import DinoV2Finetune  # ton fichier fourni

class MultiModalRegressor(nn.Module):
    def __init__(self, text_model_name='distilbert-base-uncased', freeze_dino=True):
        super().__init__()

        # --- Image encoder: DINOv2
        self.image_encoder = DinoV2Finetune(frozen=freeze_dino)
        self.image_embedding_dim = self.image_encoder.dim

        # --- Text encoder (DistilBERT)
        self.tokenizer = DistilBertTokenizer.from_pretrained(text_model_name)
        self.text_encoder = DistilBertModel.from_pretrained(text_model_name)
        self.text_embedding_dim = self.text_encoder.config.hidden_size

        # --- Fusion + MLP
        self.fc = nn.Sequential(
            nn.Linear(self.image_embedding_dim + 2 * self.text_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)  # Regression
        )

    def forward(self, image_tensor, title_texts, desc_texts):
        device = image_tensor.device

        # --- Image features
        image_feat = self.image_encoder({"image": image_tensor}).squeeze(-1)  # [B, dim]

        # --- Title features
        title_inputs = self.tokenizer(title_texts, return_tensors="pt", padding=True, truncation=True, max_length=30)
        title_inputs = {k: v.to(device) for k, v in title_inputs.items()}
        title_outputs = self.text_encoder(**title_inputs)
        title_feat = title_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # --- Description features
        desc_inputs = self.tokenizer(desc_texts, return_tensors="pt", padding=True, truncation=True, max_length=200)
        desc_inputs = {k: v.to(device) for k, v in desc_inputs.items()}
        desc_outputs = self.text_encoder(**desc_inputs)
        desc_feat = desc_outputs.last_hidden_state[:, 0, :]

        # --- Fusion & regression
        x = torch.cat([image_feat, title_feat, desc_feat], dim=1)
        return self.fc(x).squeeze(1)  # output: [B]
