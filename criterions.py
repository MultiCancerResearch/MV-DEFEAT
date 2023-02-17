import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CE_loss(nn.Module):
    def __init__(self):
        super(CE_loss, self).__init__()

    def forward(self, preds, tgts):
        loss = nn.CrossEntropyLoss()(preds, tgts)
        return loss


class Focal_loss(nn.Module):
    def __init__(self):
        super(Focal_loss, self).__init__()

    def forward(selfself, preds, tgts):
        loss = nn.CrossEntropyLoss()(preds, tgts)
        loss_exp = torch.exp(-loss)
        # alpha = 0.8, gamma =2
        focal_loss = 0.8 * (1 - loss_exp) ** 2 * loss
        return focal_loss


class EDL_CE_Loss(nn.Module):
    def __init__(self, cfg):
        self.cfg = cfg
        super(EDL_CE_Loss, self).__init__()

    # loss function
    def KL(self, alpha, c):
        # beta = torch.ones((1, c)).to('cuda')
        beta = torch.ones((1, c)).to(device)
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
        return kl

    def forward(self, pred, tgt):
        A = nn.CrossEntropyLoss()(pred, tgt)
        label = F.one_hot(tgt, num_classes=self.cfg.arch.num_classes)
        E = pred - 1
        alp = E * (1 - label) + 1
        B = self.cfg.loss.annealing_coef * self.KL(alp, self.cfg.arch.num_classes)

        return torch.mean((A + B))
