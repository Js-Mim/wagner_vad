# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

import torch

torch.has_cudnn = True
torch.set_default_tensor_type('torch.cuda.FloatTensor')

exp_settings = {
    'fs': 22050,                       # Sampling frequency
    'ft_size': 2048,                   # N-DFT size
    'hop_size': 512,                   # STFT hop-size
    'n_mel': 250,                      # Mel bands
    'classification_dim': 3,           # Dimensionality given to the classifier
    'classification_dim_ids': 3,       # Dimensionality for the output of the singer classifier
    'reg_lambda': 0.,                  # Regularization of the clustering
    'batch_size': 64,                  # Batch size
    'T': 134,                          # Analysis time-frames <-- pre-defined atm :(
    'epochs': 100,                     # Number of iterations
    'split_name': 'split_a',           # Data split identifier for saving the results under different folder
    'split_training_indx': 2,          # Integer indicating up to which opera will be used for training (#2: uses Barenboi-Kupfer and Haitink)
    'split_validation_indx': 2,        # Integer indicating the opera used for validation (#2: uses Karajan for validation)
    'd_p_length': 3,                   # Length of each data point (spectral patch) in seconds
    'drp_rate': 0.2,                   # Drop-out rate
    'learning_rate_drop': 0.8,         # Factor for decreasing the learning rate
    'learning_date_incr': 1.1,         # Factor for increasing the learning rate
    'end2end': False,                  # End2End learning flag
    'cnn_kernels': [(5, 5), (3, 3)],   # Kernel sizes
    'channels': [16, 8],               # Number of channels per layer
    't_pcen': 0.395,                   # Magic number from "Per-Channel Energy Normalization: Why and How"
                }

# EOF
