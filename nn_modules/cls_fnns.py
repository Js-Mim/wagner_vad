# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

# imports
import torch.nn as nn


class FNNClassifier(nn.Module):
    def __init__(self, cls_dim, n_in, n_out, cls_out=1):
        super(FNNClassifier, self).__init__()

        # FNN Layer A
        self.fnn_a = nn.Linear(n_in, n_out, bias=True)
        # FNN Layer B
        self.fnn_b = nn.Linear(n_out, n_out//3, bias=True)
        # FNN Layer C
        self.fnn_c = nn.Linear(n_out//3, cls_dim, bias=True)
        # Classifier
        self.fnn_cls = nn.Linear(cls_dim, cls_out, bias=False)

        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.initialize_fnn()

    def initialize_fnn(self):
        nn.init.xavier_normal(self.fnn_a.weight)
        nn.init.xavier_normal(self.fnn_b.weight)
        nn.init.xavier_normal(self.fnn_c.weight)
        nn.init.xavier_normal(self.fnn_cls.weight)
        self.fnn_a.bias.data.zero_()
        self.fnn_b.bias.data.zero_()
        self.fnn_c.bias.data.zero_()
        return None

    def forward(self, h_dec, mel_x, ld_space=False):
        mel_mask = self.relu(self.fnn_a(h_dec))
        mel_filt = mel_mask * mel_x

        cls_input = self.tanh(self.fnn_b(mel_filt))
        cls_input = self.tanh(self.fnn_c(cls_input))
        vad_prob = self.fnn_cls(cls_input)

        if not ld_space:
            return mel_filt, vad_prob
        else:
            return mel_filt, vad_prob, cls_input

# EOF
