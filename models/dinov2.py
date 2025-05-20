import torch
import torch.nn as nn


class DinoV2Finetune(nn.Module):
    def __init__(self, frozen=True, regression=False, get_tokens=False):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        self.dim = self.backbone.norm.normalized_shape[0]
        if not frozen:
            for name, param in list(self.backbone.named_parameters())[-2:]:
                param.requires_grad = True

        self.get_tokens = get_tokens

    def forward(self, x):
        if self.get_tokens:
            x = self.backbone.get_intermediate_layers(x["image"], n=1)[0]
            print(x.shape)
        else:
            x = self.backbone(x["image"])
        return x
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DinoV2Finetune(frozen=False, regression=False, get_tokens=True).to(device)
    
    image_tensor = torch.randn(2, 3, 224, 224).to(device)
    x = {"image": image_tensor}
    embeddings = model(x)
    print(embeddings.shape) 