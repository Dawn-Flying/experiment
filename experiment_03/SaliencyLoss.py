# ----------------------------
# 自定义组合损失函数
# ----------------------------
import torch
from torch import nn


class SaliencyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha  # L1 权重
        self.beta = beta    # KLD 权重
        self.l1_loss = nn.L1Loss()
        self.eps = 1e-8

    def forward(self, pred, target):
        """
        pred, target: [B, 1, H, W], 值域 [0, 1]
        """
        # L1 Loss (像素级)
        l1 = self.l1_loss(pred, target)

        # KLD Loss (需归一化为概率分布)
        # 先展平并确保非负
        pred_flat = pred.view(pred.size(0), -1)  # [B, H*W]
        target_flat = target.view(target.size(0), -1)

        # 归一化为概率分布（sum=1）
        pred_prob = pred_flat / (pred_flat.sum(dim=1, keepdim=True) + self.eps)
        target_prob = target_flat / (target_flat.sum(dim=1, keepdim=True) + self.eps)

        # 计算 KLD: sum(p * log(p/q))
        kld = (target_prob * torch.log((target_prob + self.eps) / (pred_prob + self.eps))).sum(dim=1).mean()

        return self.alpha * l1 + self.beta * kld