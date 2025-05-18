import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from models.dinov2 import DinoV2Finetune
from models.distilBert import DistilBert

class MultiModalRegressor(nn.Module):
    def __init__(self, text_model_name='distilbert-base-uncased', freeze_dino=True):
        super().__init__()

        # --- Image encoder: DINOv2
        self.image_encoder = DinoV2Finetune(frozen=freeze_dino, regression=False)
        self.image_embedding_dim = self.image_encoder.dim

        # --- Text encoder (DistilBERT)
        self.text_encoder = DistilBert(text_model_name=text_model_name)
        self.text_embedding_dim = self.text_encoder.dim

        channel_embedding_dim = 8
        self.channel_embedding = nn.Embedding(1, channel_embedding_dim)

        self.max_year = 2025
        self.min_year = 2011

        self.input_dim = self.image_embedding_dim + self.text_embedding_dim
        self.tabular_dim = 2
        self.droupout = 0.2

        # --- Fusion + MLP
        self.image_projector = nn.Sequential(
            nn.Linear(self.image_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.droupout)
        )
        self.text_projector = nn.Sequential(
            nn.Linear(self.text_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.droupout)
        )
        self.reg_head = nn.Sequential(
            nn.Dropout(self.droupout),
            nn.Linear(512+self.tabular_dim, 1),
            nn.ReLU(),
        )

        print("shape of image encoder: ", self.image_embedding_dim)
        print("shape of text encoder: ", self.text_embedding_dim)
        print("shape of fusion: ", self.input_dim)


    def forward(self, x):
        image_tensor = x["image"]
        title_texts = x["title"]
        desc_texts = x["description"]
        channel = x["channel"]
        year = x["year"]
        device = image_tensor.device

        # --- Image features
        image_feat = self.image_encoder({"image": image_tensor}).squeeze(-1)  # [B, dim]
        image_feat = self.image_projector(image_feat)

        # --- Title features
        title_feat = self.text_encoder(title_texts)
        title_feat = self.text_projector(title_feat,device)

        # --- Tabular features
        channel_feat = self.channel_embedding(channel)
        year_feat = (year - self.min_year) / (self.max_year - self.min_year)

        x = torch.cat([image_feat, title_feat, channel_feat, year_feat], dim=1)
        x = self.reg_head(x)
        
        return x

