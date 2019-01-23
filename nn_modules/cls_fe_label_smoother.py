# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import torch.nn as nn


class ClassLabelSmoother(nn.Module):
    """
        Class for building the analysis part
        of the Front-End ('Fe').
    """
    def __init__(self, ft_size=1024, hop_size=384):
        super(ClassLabelSmoother, self).__init__()

        # Parameters
        self.batch_size = None
        self.time_domain_samples = None
        self.sz = ft_size
        self.hop = hop_size

        # 1D CNN
        self.conv_smooth = nn.Conv1d(1, 1, self.sz, padding=self.sz, stride=self.hop, bias=False)

        # Initialization
        self.initialize()

    def initialize(self):
        self.conv_smooth.weight.data.fill_(1./self.sz)

    def forward(self, class_signal):
        batch_size = class_signal.size(0)
        time_domain_samples = class_signal.size(1)

        class_signal = class_signal.view(batch_size, 1, time_domain_samples)
        smooth_labels = self.conv_smooth(class_signal).transpose(1, 2)

        return smooth_labels


# EOF
