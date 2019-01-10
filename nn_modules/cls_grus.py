# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

# imports
import torch
import torch.nn as nn
from torch.autograd import Variable


class BiGRUEncoder(nn.Module):

    """ Class for bi-directional gated recurrent units.
    """

    def __init__(self, B, T, N):
        """
        Args :
            B      : (int) Batch size
            T      : (int) Length of the time-sequence.
            N      : (int) Original dimensionallity of the input.
        """
        super(BiGRUEncoder, self).__init__()
        self._B = B
        self._T = T
        self._N = N

        # Bi-GRU Encoder
        self.gruF = nn.GRUCell(self._N, self._N)
        self.gruB = nn.GRUCell(self._N, self._N)

        # Initialize the weights
        self.initialize_encoder()

    def initialize_encoder(self):
        """
            Manual weight/bias initialization.
        """
        nn.init.orthogonal(self.gruF.weight_hh)
        nn.init.xavier_normal(self.gruF.weight_ih)
        self.gruF.bias_hh.data.zero_()
        self.gruF.bias_ih.data.zero_()

        nn.init.orthogonal(self.gruB.weight_hh)
        nn.init.xavier_normal(self.gruB.weight_ih)
        self.gruB.bias_hh.data.zero_()
        self.gruB.bias_ih.data.zero_()
        print('Initialization of the Bi-GRU encoder done...')

        return None

    def forward(self, x_in):
        h_enc = Variable(torch.zeros(self._B, self._T, 2 * self._N), requires_grad=False)
        # Initialization of the hidden states
        h_t_fr = Variable(torch.zeros(self._B, self._N), requires_grad=False)
        h_t_bk = Variable(torch.zeros(self._B, self._N), requires_grad=False)
        if torch.has_cudnn:
            h_enc = h_enc.cuda()
            h_t_fr = h_t_fr.cuda()
            h_t_bk = h_t_bk.cuda()

        for t in range(self._T):
            # Bi-GRU Encoding
            h_t_fr = self.gruF((x_in[:, t, :]), h_t_fr)
            h_t_bk = self.gruB((x_in[:, self._T - t - 1, :]), h_t_bk)

            h_t = torch.cat((h_t_fr + x_in[:, t, :], h_t_bk + x_in[:, self._T - t - 1, :]), dim=1)
            h_enc[:, t, :] = h_t

        return h_enc


class GRUDecoder(nn.Module):

    """ Class for GRU decoder.
    """

    def __init__(self, B, T, N):
        """
        Args :
            B      : (int) Batch size
            T      : (int) Length of the time-sequence.
            N      : (int) Original dimensionallity of the input.
        """
        super(GRUDecoder, self).__init__()
        self._B = B
        self._T = T
        self._N = N

        # Bi-GRU Encoder
        self.gruDec = nn.GRUCell(self._N, self._N)

        # Initialize the weights
        self.initialize_decoder()

    def initialize_decoder(self):
        """
            Manual weight/bias initialization.
        """
        nn.init.orthogonal(self.gruDec.weight_hh)
        nn.init.xavier_normal(self.gruDec.weight_ih)
        self.gruDec.bias_hh.data.zero_()
        self.gruDec.bias_ih.data.zero_()

        print('Initialization of the GRU decoder done...')

        return None

    def forward(self, h_enc):
        h_dec = Variable(torch.zeros(self._B, self._T, self._N), requires_grad=False)
        # Initialization of the hidden states
        h_h_t = Variable(torch.zeros(self._B, self._N), requires_grad=False)
        if torch.has_cudnn:
            h_dec = h_dec.cuda()
            h_h_t = h_h_t.cuda()

        for t in range(self._T):
            # Bi-GRU Encoding
            h_h_t = self.gruDec((h_enc[:, t, :]), h_h_t)
            h_dec[:, t, :] = h_h_t

        return h_dec

# EOF
