import torch
import torch.nn as nn
from torchvision import models

class ResNet50(nn.Module):
    def __init__(self, frozen=True, regression=False, get_tokens=False):
        super().__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V2')
        self.get_tokens = get_tokens
        if get_tokens:
            # Garde uniquement jusqu'à layer4
            self.backbone = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                resnet.layer4
            )
        else:
            resnet.fc = nn.Identity()
            self.backbone = resnet

        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        
        if not frozen:
            for name, param in list(self.backbone.named_parameters())[-2:]:
                param.requires_grad = True
        
        self.dim = 2048
        
    def forward(self, x):
        x = self.backbone(x["image"])
        if self.get_tokens:
            x = x.flatten(2).transpose(1, 2)
        return x



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50(frozen=False, regression=False, get_tokens=True).to(device)
    
    image_tensor = torch.randn(2, 3, 224, 224).to(device)
    x = {"image": image_tensor}
    embeddings = model(x)
    print(embeddings.shape) 
    print(model.dim)  # Should print the hidden size