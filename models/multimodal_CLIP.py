import open_clip
import torch
import torch.nn as nn

class MultiModalCLIPRegressor(nn.Module):
    def __init__(self, clip_model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', channel_number=46, channel_embedding_dim=8, projection_dim=256, dropout=0.2):
        super().__init__()
        
        self.channel_embedding_dim = channel_embedding_dim

        # --- CLIP model & preprocess
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=pretrained
        )
        self.clip_model.eval()  # On désactive le training de CLIP par défaut

        # Dimensions des embeddings CLIP
        self.image_embedding_dim = self.clip_model.visual.output_dim
        self.text_embedding_dim = self.clip_model.text_projection.shape[1]

        # --- Projecteurs pour réduire la dimension
        self.image_projector = nn.Sequential(
            nn.Linear(self.image_embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.text_projector = nn.Sequential(
            nn.Linear(self.text_embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # --- Embedding pour le channel
        self.channel_embedding = nn.Embedding(channel_number, channel_embedding_dim)

        # --- Normalisation de l'année
        max_year = 2025
        min_year = 2011
        self.register_buffer("min_year", torch.tensor(min_year, dtype=torch.float32))
        self.register_buffer("max_year", torch.tensor(max_year, dtype=torch.float32))

        # --- Tête de régression
        self.reg_input_dim = 2 * projection_dim + channel_embedding_dim + 1
        self.reg_head = nn.Sequential(
            nn.Linear(self.reg_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        

        print("shape of image encoder: ", self.image_embedding_dim)
        print("shape of text encoder: ", self.text_embedding_dim)
        print("shape of channel embedding: ", self.channel_embedding_dim)
    

    def forward(self, x):
        image_tensor = x["image"]  # [B, 3, H, W]
        desc_texts = x["description"]  # liste de str
        channel = x["channel"]  # [B, 1] ou [B]
        year = x["year"]  # [B, 1] ou [B]

        device = image_tensor.device

        # --- Image features via CLIP
        with torch.no_grad():
            image_feat = self.clip_model.encode_image(image_tensor)
        image_feat = self.image_projector(image_feat)

        # --- Text features via CLIP
        with torch.no_grad():
            text_tokens = open_clip.tokenize(desc_texts).to(device)
            text_feat = self.clip_model.encode_text(text_tokens)
        text_feat = self.text_projector(text_feat)

         # --- Channel embedding
        # channel: [B, 1] ou [B], il faut [B]
        if channel.ndim == 2 and channel.shape[1] == 1:
            channel = channel.squeeze(1)
        channel_feat = self.channel_embedding(channel)  # [B, channel_embedding_dim]

        # --- Year normalization
        year = year.float()
        if year.ndim == 2 and year.shape[1] == 1:
            year = year.squeeze(1)
        year_feat = ((year - self.min_year) / (self.max_year - self.min_year)).unsqueeze(1)  # [B, 1]

        # --- Fusion & régression
        # S'assurer que toutes les shapes sont [B, D]
        x_cat = torch.cat([image_feat, text_feat, channel_feat, year_feat], dim=1)
        out = self.reg_head(x_cat)
        return out