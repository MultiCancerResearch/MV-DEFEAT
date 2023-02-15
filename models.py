import os
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class MV_DEFEAT_ddsm(nn.Module):
    def __init__(self, cfg):
        super(MV_DEFEAT_ddsm, self).__init__()
        self.cfg = cfg

        self.model = timm.create_model(self.cfg.arch.model, pretrained=self.cfg.arch.pretrained)
        if self.cfg.arch.model == 'resnet50':
            self.linear_layers = self.model.fc.in_features
        #if self.cfg.arch.model == 'densenet161':
        #    self.linear_layers = self.model.classifier.in_features
        modules = list(self.model.children())[:-1]
        self.model = nn.Sequential(*modules)

        self.fc = nn.Sequential(
            nn.Linear(self.linear_layers, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.cfg.arch.num_classes)
        )

    def DS_Combin_two(self, alpha1, alpha2):
        # Calculate the merger of two DS evidences
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.cfg.arch.num_classes / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, self.cfg.arch.num_classes, 1), b[1].view(-1, 1, self.cfg.arch.num_classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate K
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
        K = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - K).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - K).view(-1, 1).expand(u[0].shape))
        # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

        # calculate new S
        S_a = self.cfg.arch.num_classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a, u_a

    def forward(self, lcc, rcc, lmlo, rmlo):
        out_lmlo = self.model(lmlo)
        out_lmlo = out_lmlo.view(out_lmlo.size(0), -1)
        out_lmlo = self.fc(out_lmlo)
        out_lmlo_evidence = F.softplus(out_lmlo)
        out_lmlo_alpha = out_lmlo_evidence + 1

        out_lcc = self.model(lcc)
        out_lcc = out_lcc.view(out_lcc.size(0), -1)
        out_lcc = self.fc(out_lcc)
        out_lcc_evidence = F.softplus(out_lcc)
        out_lcc_alpha = out_lcc_evidence + 1

        out_rmlo = self.model(rmlo)
        out_rmlo = out_rmlo.view(out_rmlo.size(0), -1)
        out_rmlo = self.fc(out_rmlo)
        out_rmlo_evidence = F.softplus(out_rmlo)
        out_rmlo_alpha = out_rmlo_evidence + 1

        out_rcc = self.model(rcc)
        out_rcc = out_rcc.view(out_rcc.size(0), -1)
        out_rcc = self.fc(out_rcc)
        out_rcc_evidence = F.softplus(out_rcc)
        out_rcc_alpha = out_rcc_evidence + 1

        if self.cfg.data.analysis == 'bilateral_cc':
            out, u = self.DS_Combin_two(out_lcc_alpha, out_rcc_alpha)
            out = nn.Softplus()(out)
            return out

        elif self.cfg.data.analysis == 'bilateral_mlo':
            out, u = self.DS_Combin_two(out_lmlo_alpha, out_rmlo_alpha)
            out = nn.Softplus()(out)
            return out

        elif self.cfg.data.analysis == 'ipsilateral_left':
            out, u = self.DS_Combin_two(out_lcc_alpha, out_lmlo_alpha)
            out = nn.Softplus()(out)
            return out

        elif self.cfg.data.analysis == 'ipsilateral_right':
            out, u = self.DS_Combin_two(out_rcc_alpha, out_rmlo_alpha)
            out = nn.Softplus()(out)
            return out

