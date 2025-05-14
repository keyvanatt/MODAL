import torch
import torch.nn as nn


class DinoV2Finetune(nn.Module):
    def __init__(self, frozen=False, regression=True):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
        self.backbone.head = nn.Identity()
        self.dim = self.backbone.norm.normalized_shape[0]
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.regression_head = nn.Sequential(
            nn.Linear(self.backbone.norm.normalized_shape[0], 1),
            nn.ReLU(),
        )
        self.regression = regression
        self.accuracy = None
        self.loss = None

    def forward(self, x):
        x = self.backbone(x["image"])
        if self.regression:
            x = self.regression_head(x)
        return x
