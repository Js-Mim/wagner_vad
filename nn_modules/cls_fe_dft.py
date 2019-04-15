# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

# imports
import numpy as np
import torch
import torch.nn as nn
import librosa
from scipy import signal as sig


class Analysis(nn.Module):
    """
        Class for building the analysis part
        of the Front-End ('Fe').
    """
    def __init__(self, ft_size=1024, hop_size=384):
        super(Analysis, self).__init__()

        # Parameters
        self.batch_size = None
        self.time_domain_samples = None
        self.sz = ft_size
        self.hop = hop_size
        self.half_N = int(self.sz/2. + 1)

        # Analysis 1D CNN
        self.conv_analysis_real = nn.Conv1d(1, self.sz, self.sz,
                                            padding=self.sz, stride=self.hop, bias=False)

        self.conv_analysis_imag = nn.Conv1d(1, self.sz, self.sz,
                                            padding=self.sz, stride=self.hop, bias=False)

        # Custom Initialization with Fourier matrix
        self.initialize()

    def initialize(self):
        f_matrix = np.fft.fft(np.eye(self.sz), norm='ortho')
        w = sig.hamming(self.sz)

        f_matrix_real = (np.real(f_matrix) * w).astype(np.float32)
        f_matrix_imag = (np.imag(f_matrix) * w).astype(np.float32)

        if torch.has_cudnn:
            self.conv_analysis_real.weight.data.copy_(torch.from_numpy(f_matrix_real[:, None, :]).cuda())
            self.conv_analysis_imag.weight.data.copy_(torch.from_numpy(f_matrix_imag[:, None, :]).cuda())
        else:
            self.conv_analysis_real.weight.data.copy_(torch.from_numpy(f_matrix_real[:, None, :]))
            self.conv_analysis_imag.weight.data.copy_(torch.from_numpy(f_matrix_imag[:, None, :]))

    def forward(self, wave_form):
        batch_size = wave_form.size(0)
        time_domain_samples = wave_form.size(1)

        wave_form = wave_form.view(batch_size, 1, time_domain_samples)
        an_real = self.conv_analysis_real(wave_form).transpose(1, 2)[:, :, :self.half_N]
        an_imag = self.conv_analysis_imag(wave_form).transpose(1, 2)[:, :, :self.half_N]

        return an_real, an_imag


class MelFilterbank(nn.Module):
    """
        Class for building the analysis part
        of the Front-End ('Fe').
    """
    def __init__(self, fs, n_mel, n_fft):
        super(MelFilterbank, self).__init__()

        # Parameters
        self.fs = fs
        self.n_mel = n_mel
        self.n_fft = n_fft
        self.half_N = int(n_fft/2. + 1)

        # Mel Layer
        self.mel_analysis = nn.Linear(self.half_N, n_mel, bias=False)

        # Custom Initialization with Fourier matrix
        self.initialize()

    def initialize(self):
        mel_matrix = librosa.filters.mel(self.fs, self.n_fft, self.n_mel, norm=None)

        if torch.has_cudnn:
            self.mel_analysis.weight.data.copy_(torch.from_numpy(mel_matrix).cuda())
        else:
            self.mel_analysis.weight.data.copy_(torch.from_numpy(mel_matrix))

    def forward(self, mX):
        mel_mx = self.mel_analysis(mX)
        return mel_mx


# EOF
