# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

# imports
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
        A VGG-like block of 2D convolutional layers.
        A zero-mean update is performed according to:
        Jan Schlüter and Bernhard Lehner, "Zero-mean convolutions for level-invariant singing voice detection"
    """
    def __init__(self, kernels=[(3, 3), (3, 3)], channels=[128, 64]):
        super(ConvBlock, self).__init__()

        self.epsilon = 1e-4

        # Conv layers
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=channels[0],
                               kernel_size=kernels[0], stride=(1, 1),
                               padding=(kernels[0][0]//2, kernels[0][1]//2), bias=False)

        self.conv2 = nn.Conv2d(in_channels=channels[0],
                               out_channels=channels[1],
                               kernel_size=kernels[1], stride=(1, 1),
                               padding=(kernels[1][0]//2, kernels[1][1]//2), bias=False)

        self.ffn = nn.Linear(in_features=channels[1], out_features=1, bias=False)

        self.relu = nn.ReLU()

        self.initialize_convs()

    def initialize_convs(self):
        torch.nn.init.xavier_normal(self.conv1.weight)
        torch.nn.init.xavier_normal(self.conv2.weight)
        torch.nn.init.xavier_normal(self.ffn.weight)

    def forward(self, mel_in):
        x = torch.log(mel_in.unsqueeze(1) + self.epsilon)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(self.ffn(x.permute(0, 2, 3, 1)).squeeze(-1))
        return x

    def zero_mean(self):
        self.conv1.weight.data = self.conv1.weight.data - \
                        torch.mean(torch.mean(self.conv1.weight.data, dim=3,
                                              keepdim=True), dim=2, keepdim=True)
        self.conv2.weight.data = self.conv2.weight.data - \
                        torch.mean(torch.mean(torch.mean(self.conv2.weight.data, dim=3, keepdim=True),
                                              dim=2, keepdim=True), dim=1, keepdim=True)

# EOF
