# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothLoss(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.1, loss_weight: float = 1.0, reduction: str = 'mean'):
        super(LabelSmoothLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.smooth_factor = smoothing / (num_classes - 1)
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        target = F.one_hot(target, num_classes=self.num_classes).float()
        target = target * self.confidence + self.smooth_factor

        log_prob = F.log_softmax(pred, dim=-1)
        loss = -torch.sum(target * log_prob, dim=-1)

        # Apply reduction
        if self.reduction == 'mean':
            return self.loss_weight * loss.mean()
        elif self.reduction == 'sum':
            return self.loss_weight * loss.sum()
        else:  # 'none'
            return self.loss_weight * loss
