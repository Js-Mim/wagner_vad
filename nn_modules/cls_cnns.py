# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

# imports
import torch
import torch.nn as nn


class SchModel(nn.Module):
    """
        A pytorch implementation of the architecture presented in:
        Jan Schlüter and Bernhard Lehner, "Zero-mean convolutions for level-invariant singing voice detection"
    """

    def __init__(self):
        super(SchModel, self).__init__()

        self.epsilon = 1e-6
        self.lrelu = nn.LeakyReLU(1e-2)
        self.dropout = nn.Dropout(0.5)

        # Conv layers
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=32,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=128,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128,
                               out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=(3, 18), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn5 = nn.BatchNorm2d(128)
        self.ffn1 = nn.Linear(in_features=128, out_features=1, bias=True)
        self.ffn2 = nn.Linear(in_features=68, out_features=32, bias=True)
        self.ffn3 = nn.Linear(in_features=32, out_features=1, bias=False)

        self.initialize()

    def initialize(self):
        torch.nn.init.xavier_normal(self.conv1.weight)
        torch.nn.init.xavier_normal(self.conv2.weight)
        torch.nn.init.xavier_normal(self.conv3.weight)
        torch.nn.init.xavier_normal(self.conv4.weight)
        torch.nn.init.xavier_normal(self.conv5.weight)
        torch.nn.init.xavier_normal(self.ffn1.weight)
        torch.nn.init.xavier_normal(self.ffn2.weight)
        torch.nn.init.xavier_normal(self.ffn3.weight)
        self.ffn1.bias.data.zero_()
        self.ffn2.bias.data.zero_()

    def forward(self, mel_in):
        x = torch.log(mel_in.unsqueeze(1) + self.epsilon)
        x = self.lrelu(self.bn1(self.conv1(x)))
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, kernel_size=(1, 3))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = self.lrelu(self.bn4(self.conv4(x)))
        x = self.lrelu(self.bn5(self.conv5(x)))
        x = self.lrelu(self.ffn1(x.permute(0, 2, 3, 1)).squeeze(-1))
        x = self.dropout(x)
        x = self.lrelu(self.ffn2(x))
        x = self.ffn3(x)
        return x

    def zero_mean(self):
        self.conv1.weight.data = self.conv1.weight.data - \
                                 torch.mean(torch.mean(self.conv1.weight.data, dim=3,
                                                       keepdim=True), dim=2, keepdim=True)
        self.conv2.weight.data = self.conv2.weight.data - \
                                 torch.mean(torch.mean(torch.mean(self.conv2.weight.data, dim=3, keepdim=True),
                                                       dim=2, keepdim=True), dim=1, keepdim=True)

        self.conv3.weight.data = self.conv3.weight.data - \
                                 torch.mean(torch.mean(torch.mean(self.conv3.weight.data, dim=3, keepdim=True),
                                                       dim=2, keepdim=True), dim=1, keepdim=True)

        self.conv4.weight.data = self.conv4.weight.data - \
                                 torch.mean(torch.mean(torch.mean(self.conv4.weight.data, dim=3, keepdim=True),
                                                       dim=2, keepdim=True), dim=1, keepdim=True)

        self.conv5.weight.data = self.conv5.weight.data - \
                                 torch.mean(torch.mean(torch.mean(self.conv5.weight.data, dim=3, keepdim=True),
                                                       dim=2, keepdim=True), dim=1, keepdim=True)

# EOF
