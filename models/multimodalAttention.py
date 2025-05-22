import torch
import torch.nn as nn
from models.dinov2 import DinoV2Finetune
from models.distilBert import DistilBertEncoder

class MultiModalAttentionRegressor(nn.Module):
    def __init__(self, text_model_name='distilbert-base-multilingual-cased', freeze_dino=True):
        super().__init__()

        # --- Image encoder: DINOv2
        self.image_encoder = DinoV2Finetune(frozen=freeze_dino, regression=False,get_tokens=True)
        self.image_embedding_dim = self.image_encoder.dim

        # --- Text encoder (DistilBERT)
        self.text_encoder = DistilBertEncoder(model_name=text_model_name, pool=False)
        self.text_embedding_dim = self.text_encoder.dim

        assert self.image_embedding_dim == self.text_embedding_dim, "Image and text embedding dimensions must match."

        self.channel_embedding_dim = 8
        self.channel_number = 46
        self.channel_embedding = nn.Embedding(self.channel_number, self.channel_embedding_dim)

        max_year = 2025
        min_year = 2011
        self.register_buffer("min_year", torch.tensor(min_year, dtype=torch.float32))
        self.register_buffer("max_year", torch.tensor(max_year, dtype=torch.float32))

        self.tabular_dim = 2
        self.droupout = 0.2


        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.image_embedding_dim,
            num_heads=8,
            batch_first=True
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.project_dim = 512
        self.reg_input_dim = self.project_dim+self.channel_embedding_dim+1
        self.projector = nn.Sequential(
            nn.Linear(self.image_embedding_dim, self.project_dim),
            nn.ReLU(),
            nn.Dropout(self.droupout),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(self.reg_input_dim, self.reg_input_dim//2),
            nn.ReLU(),
            nn.Dropout(self.droupout),
            nn.Linear(self.reg_input_dim//2, 1),
            nn.ReLU(),
            nn.Dropout(self.droupout),
        )
            
        

        print("shape of image encoder: ", self.image_embedding_dim)
        print("shape of text encoder: ", self.text_embedding_dim)
        print("shape of channel embedding: ", self.channel_embedding_dim)

    def forward(self, x):

        image_tensor = x["image"]
        title_texts = x["title"]
        desc_texts = x["description"]
        channel = x["channel"]
        year = x["year"]

        # --- Image features
        image_tokens = self.image_encoder({"image": image_tensor})

        # --- Title features
        text_tokens = self.text_encoder(title_texts)


        # text_tokens: (batch, seq_len_text, embed_dim) -> Q
        # image_tokens: (batch, seq_len_img, embed_dim) -> K, V
        # nn.MultiheadAttention expects (batch, seq, embed_dim) with batch_first=True
        attn_output, _ = self.cross_attn(query=text_tokens, key=image_tokens, value=image_tokens)
        x = attn_output.transpose(1, 2)  # (batch, embed_dim, seq_len_text)
        x = self.pool(x).squeeze(-1)  # (batch, embed_dim)

        channel_feat = self.channel_embedding(channel.squeeze(1))  # [B, dim]
        year = year.float()
        year_feat = (year - self.min_year) / (self.max_year - self.min_year)
        x = self.projector(x)  # (batch, project_dim)
        x = torch.cat([x, channel_feat, year_feat], dim=1)
        x = self.reg_head(x)  # (batch, 1)
        return x
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalAttentionRegressor(text_model_name='distilbert-base-multilingual-cased', freeze_dino=False).to(device)
    
    image_tensor = torch.randn(2, 3, 224, 224).to(device)
    title_texts = ["Bonjour", "Hello"]
    desc_texts = ["Description 1", "Description 2"]
    channel = torch.randint(0, 46, (2, 1)).to(device)
    year = torch.tensor([[2015], [2020]]).to(device)

    x = {
        "image": image_tensor,
        "title": title_texts,
        "description": desc_texts,
        "channel": channel,
        "year": year
    }
    
    embeddings = model(x)
    print(embeddings.shape)  # Should print: (2, 1)
