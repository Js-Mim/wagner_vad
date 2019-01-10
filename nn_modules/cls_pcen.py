# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

# imports
import torch
import torch.nn as nn


class PCEN(nn.Module):
    """
        A learnable per-channel energy normalization operation.
    """
    def __init__(self, N, T, sr=22050, t=0.395, hop_size=512):
        super(PCEN, self).__init__()
        self.N = N
        self.T = T
        self.log_alpha = nn.Parameter((torch.randn(self.N) * 0.1 + 1.).log_())
        self.log_delta = nn.Parameter((torch.randn(self.N) * 0.1 + 2.).log_())
        self.log_rho = nn.Parameter((torch.randn(self.N) * 0.1 + 0.6).log_())
        self.log_sigma = nn.Parameter(1. - torch.exp(-torch.ones(1)*float(hop_size)/(t * sr)))
        self.eps = 1e-1

    def forward(self, x):
        alpha = self.log_alpha.expand_as(x).exp()
        delta = self.log_delta.expand_as(x).exp()
        rho = self.log_rho.expand_as(x).exp()
        sigma = self.log_sigma.exp()

        m = torch.autograd.Variable(torch.zeros(x.size()))
        for t_index in range(self.T):
            m_prv = m[:, t_index-1, :].clone()
            m[:, t_index, :] = sigma * x[:, t_index, :] + (1. - sigma) * m_prv

        pcen_out = (x / ((m + self.eps) ** alpha) + delta) ** rho - delta ** rho

        return pcen_out


class PCENlr(nn.Module):
    """
        A Low-rank version for per-channel energy normalization.
    """
    def __init__(self, N, T):
        super(PCENlr, self).__init__()
        self.N = N
        self.T = T
        self.lr_enc = nn.Linear(self.T, 1, bias=False)
        self.lr_dec = nn.Linear(1, self.T, bias=False)
        self.log_alpha = nn.Parameter((torch.randn(self.N) * 0.1 + 1.).log_())
        self.log_delta = nn.Parameter((torch.randn(self.N) * 0.1 + 2.).log_())
        self.log_rho = nn.Parameter((torch.randn(self.N) * 0.1 + 0.6).log_())
        self.relu = nn.ReLU()
        self.eps = 1e-1

        self.initialize_parameters()

    def initialize_parameters(self):
        torch.nn.init.xavier_normal(self.lr_enc.weight)

    def forward(self, x):
        alpha = self.log_alpha.expand_as(x).exp()
        delta = self.log_delta.expand_as(x).exp()
        rho = self.log_rho.expand_as(x).exp()

        m = self.relu(self.lr_dec(self.lr_enc(x.permute(0, 2, 1))).permute(0, 2, 1) + x)
        pcen_out = (x / ((m + self.eps) ** alpha) + delta) ** rho - delta ** rho

        return pcen_out

# EOF
