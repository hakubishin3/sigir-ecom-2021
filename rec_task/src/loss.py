import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
    """
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class ClassBalancedLoss(nn.Module):
    def __init__(self, samples_per_cls, no_of_classes, loss_type, beta, gamma):
        super(ClassBalancedLoss, self).__init__()
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets):
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights[:2] = 1e-8   # nan and pad
        weights = torch.tensor(weights).float().to("cuda")
        weights = weights.unsqueeze(0).repeat(targets.shape[0], 1)

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-1 * BCE_loss)
        F_loss = weights * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)
