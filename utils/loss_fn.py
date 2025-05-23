import torch
from torch import nn


class RMSLELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSLELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        # Ensure the predictions and targets are non-negative
        # -- added a clamp to avoid log(0) and ReLU in the
        y_pred = torch.clamp(y_pred, min=0)
        y_true = torch.clamp(y_true, min=0)

        # Compute the RMSLE
        log_pred = torch.log1p(y_pred)
        log_true = torch.log1p(y_true)
        loss = torch.sqrt(torch.mean((log_pred - log_true) ** 2))

        return loss


class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Ensure the predictions and targets are non-negative
        # -- added a clamp to avoid log(0) and ReLU in the
        y_pred = torch.clamp(y_pred, min=0)
        y_true = torch.clamp(y_true, min=0)

        # Compute the RMSLE
        log_pred = torch.log1p(y_pred)
        log_true = torch.log1p(y_true)
        loss = torch.mean((log_pred - log_true) ** 2)

        return loss



class WEIGHTED_MSELoss(nn.Module):
    def __init__(self):
        super(WEIGHTED_MSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Ensure the predictions and targets are non-negative
        # -- added a clamp to avoid log(0) and ReLU in the
        y_pred = torch.clamp(y_pred, min=0)
        y_true = torch.clamp(y_true, min=0)

        weights =  1.0 + torch.clamp(y_true-10, min=0) / 10.0

        # Compute the weighted MSE
        loss = torch.mean(weights * (y_pred - y_true) ** 2)

        return loss