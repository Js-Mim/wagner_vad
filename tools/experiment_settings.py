# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

exp_settings = {
    'fs': 22050,                       # Sampling frequency
    'ft_size': 2048,                   # N-DFT size
    'hop_size': 512,                   # STFT hop-size
    'n_mel': 250,                      # Mel bands
    'classification_dim': 3,           # Dimensionality for classification
    'batch_size': 64,                  # Batch size
    'T': 134,                          # Analysis time-frames <-- pre-defined atm :(
    'epochs': 150,                     # Number of iterations
    'd_p_length': 3,                   # Length of each data point (spectral patch) in seconds
    'drp_rate': 0.2,                   # Drop-out rate -->                 # 0.5 for reproducing the results for the GRU
    'learning_rate_drop': 0.8,         # Factor for decreasing the learning rate
    'learning_date_incr': 1.1,         # Factor for increasing the learning rate
    'end2end': False,                  # End2End learning flag
    'cnn_kernels': [(5, 5), (3, 3)],   # Kernel sizes
    'channels': [16, 8],               # Number of channels per layer
    't_pcen': 0.395,                   # Magic number from "Per-Channel Energy Normalization: Why and How"
                }

# EOF
