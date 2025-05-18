import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from models.dinov2 import DinoV2Finetune

class MultiModalRegressor(nn.Module):
    def __init__(self, text_model_name='distilbert-base-uncased', freeze_dino=True):
        super().__init__()

        # --- Image encoder: DINOv2
        self.image_encoder = DinoV2Finetune(frozen=freeze_dino, regression=False)
        self.image_embedding_dim = self.image_encoder.dim

        # --- Text encoder (DistilBERT)
        self.tokenizer = DistilBertTokenizer.from_pretrained(text_model_name)
        self.text_encoder = DistilBertModel.from_pretrained(text_model_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        # Unfreeze the last two layers of the backbone
        for name, param in list(self.text_encoder.named_parameters())[-2:]:
            param.requires_grad = True
        self.text_embedding_dim = self.text_encoder.config.hidden_size

        self.input_dim = self.image_embedding_dim + self.text_embedding_dim
        self.tabular_dim = 2
        self.droupout = 0.2

        # --- Fusion + MLP
        self.img_projector = nn.Sequential(
            nn.Linear(self.image_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.droupout)
        )
        self.text_projector = nn.Sequential(
            nn.Linear(self.text_embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.droupout)
        )
        self.fc = nn.Sequential(
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

        # --- Title features
        title_inputs = self.tokenizer(title_texts, return_tensors="pt", padding=True, truncation=True, max_length=30)
        title_inputs = {k: v.to(device) for k, v in title_inputs.items()}
        title_outputs = self.text_encoder(**title_inputs)
        title_feat = title_outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # --- Description features
        """desc_inputs = self.tokenizer(desc_texts, return_tensors="pt", padding=True, truncation=True, max_length=200)
        desc_inputs = {k: v.to(device) for k, v in desc_inputs.items()}
        desc_outputs = self.text_encoder(**desc_inputs)
        desc_feat = desc_outputs.last_hidden_state[:, 0, :]"""

        # --- Fusion & regression
        image_feat = self.img_projector(image_feat)
        title_feat = self.text_projector(title_feat)
        x = torch.cat([image_feat, title_feat, channel.float(), year.float()], dim=1)
        x = self.fc(x)
        return x

