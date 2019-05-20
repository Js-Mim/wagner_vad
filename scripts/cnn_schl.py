# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

"""
    Supervised model for singing voice activity detection.
    Ingredients: 1D-Conv for STFT, 1FFN-Mel,
                 1FFN-PCEN, Dropout, Bi-GRU Enc.,
                 GRU Dec., FFN-Skip-connections,
                 Max-pooling, FFN-Sigmoid
"""

# imports
import numpy as np
import torch
import os
from sklearn.metrics import precision_recall_fscore_support as prf
from tqdm import tqdm
from nn_modules import cls_fe_dft, cls_cnns, cls_fe_label_smoother
from tools import helpers, visualize
from torch.optim.lr_scheduler import StepLR
from tools.experiment_settings import exp_settings
from torch.nn import functional as F


def build_model(flag='training'):
    print('--- Building Model---')
    # Front-ends for feature extraction
    dft_analysis = cls_fe_dft.Analysis(ft_size=exp_settings['ft_size'],
                                       hop_size=exp_settings['hop_size'])

    mel_analysis = cls_fe_dft.MelFilterbank(exp_settings['fs'], exp_settings['n_mel'],
                                            exp_settings['ft_size'])
    # Zero mean CNNs baseline
    schl_model = cls_cnns.SchModel()

    # Label smoother
    label_conv = cls_fe_label_smoother.ClassLabelSmoother(ft_size=exp_settings['ft_size'],
                                                          hop_size=exp_settings['hop_size'])

    if flag == 'testing':
        print('--- Loading Model ---')
        schl_model.load_state_dict(torch.load(os.path.join('results', exp_settings['split_name'],
                                              'schl_model.pytorch'),
                                              map_location=lambda storage, location: storage))

    if flag == 'training':
        dft_analysis = dft_analysis.cuda()
        mel_analysis = mel_analysis.cuda()
        schl_model = schl_model.cuda()
        schl_model = schl_model.train()
    else:
        schl_model = schl_model.eval()
        if not torch.has_cudnn:
            dft_analysis = dft_analysis.cpu()
            mel_analysis = mel_analysis.cpu()
            schl_model = schl_model.cpu()

    return dft_analysis, mel_analysis, schl_model, label_conv


def perform_training():
    # Check if saving path exists
    if not (os.path.isdir(os.path.join("results/" + exp_settings['split_name']))):
        print('Saving directory was not found... Creating a new folder to store the results!')
        os.makedirs(os.path.join("results/" + exp_settings['split_name']))

    # Get data dictionary
    data_dict = helpers.csv_to_dict(training=True)
    training_keys = sorted(list(data_dict.keys()))[0:exp_settings['split_training_indx']]
    print('Training on: ' + " ".join(training_keys))

    # Get data
    x, y, _ = helpers.fetch_data(data_dict, training_keys)
    x *= 0.99 / np.max(np.abs(x))

    d_p_length_samples = exp_settings['d_p_length'] * exp_settings['fs']  # Length in samples

    # Initialize NN modules
    win_viz, _ = visualize.init_visdom()    # Web loss plotting
    dft_analysis, mel_analysis, schl_model, label_smoother = build_model(flag='training')

    # Criterion
    bce_func = torch.nn.BCEWithLogitsLoss(size_average=True)

    # Initialize optimizer and add the parameters that will be updated
    if exp_settings['end2end']:
        parameters_list = list(dft_analysis.parameters()) + list(mel_analysis.parameters()) +\
                          list(schl_model.parameters())
    else:
        parameters_list = list(schl_model.parameters())

    optimizer = torch.optim.Adam(parameters_list, lr=1e-4)
    scheduler_n = StepLR(optimizer, 1, gamma=exp_settings['learning_rate_drop'])
    scheduler_p = StepLR(optimizer, 1, gamma=exp_settings['learning_date_incr'])

    # Start of the training
    batch_indx = 0
    number_of_data_points = len(x) // d_p_length_samples
    prv_cls_error = 100.
    best_error = 50.
    for epoch in range(1, exp_settings['epochs'] + 1):
        # Validation
        if not epoch == 1:
            cls_err = perform_validation([dft_analysis, mel_analysis, schl_model, label_smoother])

            if prv_cls_error - cls_err > 0:
                # Increase learning rate
                scheduler_p.step()
                if cls_err < best_error:
                    # Update best error
                    best_error = cls_err
                    print('--- Saving Model ---')
                    torch.save(schl_model.state_dict(), os.path.join('results', exp_settings['split_name'],
                                                                     'schl_model.pytorch'))

            else:
                # Decrease learning rate
                scheduler_n.step()

            # Update classification error
            prv_cls_error = cls_err

        # Shuffle between sequences
        shuffled_data_points = np.random.permutation(np.arange(0, number_of_data_points))

        # Constructing batches
        available_batches = len(shuffled_data_points)//exp_settings['batch_size']
        for batch in tqdm(range(available_batches)):
            x_d_p, y_d_p = helpers.gimme_batches(batch, shuffled_data_points, x, y)

            x_cuda = torch.autograd.Variable(torch.from_numpy(x_d_p).cuda(), requires_grad=False).float().detach()
            y_cuda = torch.autograd.Variable(torch.from_numpy(y_d_p).cuda(), requires_grad=False).float().detach()

            # Forward analysis pass: Input data
            x_real, x_imag = dft_analysis.forward(x_cuda)
            # Magnitude computation
            mag = torch.sqrt(x_real.pow(2) + x_imag.pow(2))

            # Mel analysis
            mel_mag = torch.autograd.Variable(mel_analysis.forward(mag).data, requires_grad=True).cuda()

            # CNN-based classifer
            vad_prob = schl_model.forward(mel_mag)

            # Target data preparation
            y_true = label_smoother.forward(y_cuda).detach()
            vad_true = torch.autograd.Variable(y_true.data, requires_grad=True).cuda()

            # Loss
            loss = bce_func(vad_prob, vad_true)

            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schl_model.zero_mean()

            # Update graph
            win_viz = visualize.viz.line(X=np.arange(batch_indx, batch_indx + 1),
                                         Y=np.reshape(loss.data[0], (1,)),
                                         win=win_viz, update='append')
            batch_indx += 1

    return None


def perform_validation(nn_list):
    print('---Performing Validation---')
    d_p_length_samples = exp_settings['d_p_length'] * exp_settings['fs']
    # Get data dictionary
    data_dict = helpers.csv_to_dict(training=True)
    keys = sorted(list(data_dict.keys()))
    validation_key = keys[exp_settings['split_validation_indx']]
    print('Validating on: ' + " ".join(validation_key))
    # Get data
    x, y, _ = helpers.fetch_data(data_dict, validation_key)
    x *= 0.99 / np.max(np.abs(x))

    sigmoid = torch.nn.Sigmoid()  # Label helper!

    # Constructing batches
    number_of_data_points = len(x) // d_p_length_samples
    available_batches = number_of_data_points // exp_settings['batch_size']
    data_points = np.arange(0, number_of_data_points)
    for batch in tqdm(range(available_batches)):
        x_d_p, y_d_p = helpers.gimme_batches(batch, data_points, x, y)

        x_cuda = torch.autograd.Variable(torch.from_numpy(x_d_p).cuda(), requires_grad=False).float().detach()
        y_cuda = torch.autograd.Variable(torch.from_numpy(y_d_p).cuda(), requires_grad=False).float().detach()

        # Forward analysis pass: Input data
        x_real, x_imag = nn_list[0].forward(x_cuda)
        # Magnitude computation
        mag = torch.sqrt(x_real.pow(2) + x_imag.pow(2))
        # Mel analysis
        mel_mag = torch.autograd.Variable(nn_list[1].forward(mag).data, requires_grad=True)

        # Main model
        vad_prob = nn_list[2].forward(mel_mag)
        vad_prob = sigmoid(vad_prob)
        vad_prob = vad_prob.gt(0.51).float().data.cpu().numpy()[:, :, 0]\
            .reshape(exp_settings['batch_size']*exp_settings['T'], 1)

        # Target data preparation
        y_true = nn_list[3].forward(y_cuda).detach()[:, :, 0]
        vad_true = y_true.gt(0.51).float().data.cpu().numpy().reshape(exp_settings['batch_size']*exp_settings['T'], 1)

        if batch == 0:
            out_prob = vad_prob
            out_true_prob = vad_true
        else:
            out_prob = np.vstack((out_prob, vad_prob))
            out_true_prob = np.vstack((out_true_prob, vad_true))

    res = prf(out_true_prob, out_prob, average='binary')
    cls_error = np.sum(np.abs(out_true_prob - out_prob))/len(out_true_prob) * 100.

    print('Precision: %2f' % res[0])
    print('Recall: %2f' % res[1])
    print('Fscore: %2f' % res[2])
    print('Error: %2f' % cls_error)

    return cls_error


def perform_testing():
    print('---Performing Evaluation---')
    nn_list = list(build_model(flag='testing'))
    data_dict = helpers.csv_to_dict(training=False)
    keys = list(data_dict.keys())
    testing_key = keys[0]
    print('Testing on: ' + ' '.join(testing_key))
    # Get data
    x, y, fs = helpers.fetch_data(data_dict, testing_key)
    x *= 0.99 / np.max(np.abs(x))

    sigmoid = torch.nn.Sigmoid()  # Label helper!
    d_p_length_samples = exp_settings['d_p_length'] * exp_settings['fs']  # Length in samples

    number_of_data_points = len(x) // d_p_length_samples
    for data_point in tqdm(range(number_of_data_points)):
        # Generate data
        x_d_p = x[data_point*d_p_length_samples:(data_point+1)*d_p_length_samples]
        y_d_p = y[data_point*d_p_length_samples:(data_point+1)*d_p_length_samples]

        # Reshape data
        x_d_p = x_d_p.reshape(1, d_p_length_samples)
        y_d_p = y_d_p.reshape(1, d_p_length_samples)
        x_cuda = torch.autograd.Variable(torch.from_numpy(x_d_p), requires_grad=False).float().detach()
        y_cuda = torch.autograd.Variable(torch.from_numpy(y_d_p), requires_grad=False).float().detach()
        if torch.has_cudnn:
            x_cuda = x_cuda.cuda()
            y_cuda = y_cuda.cuda()

        # Forward analysis pass: Input data
        x_real, x_imag = nn_list[0].forward(x_cuda)
        # Magnitude computation
        mag = torch.norm(torch.cat((x_real, x_imag), 0), 2, dim=0).unsqueeze(0)
        # Mel analysis
        mel_mag = torch.autograd.Variable(nn_list[1].forward(mag).data, requires_grad=False)

        # Learned normalization
        vad_prob = sigmoid(nn_list[2].forward(mel_mag))

        # Up-sample the labels to the time-domain
        vad_prob_us = F.conv_transpose1d(torch.transpose(vad_prob, 1, 2),
                                         nn_list[3].conv_smooth.weight * exp_settings['hop_size'],
                                         None, exp_settings['hop_size'],
                                         padding=exp_settings['ft_size'] - exp_settings['hop_size'],
                                         dilation=1, groups=1).gt(0.50).float().data.cpu().numpy()

        vad_true = y_cuda.float().data.cpu().numpy()[0, :]

        vad_prob_us = vad_prob_us[0, 0, exp_settings['hop_size']:exp_settings['hop_size'] +
                                                                 exp_settings['fs'] * exp_settings['d_p_length']]

        if data_point == 0:
            out_prob = vad_prob_us
            out_true_prob = vad_true
        else:
            out_prob = np.hstack((out_prob, vad_prob_us))
            out_true_prob = np.hstack((out_true_prob, vad_true))

    res = prf(out_true_prob, out_prob, average='binary')
    cls_error = np.sum(np.abs(out_true_prob - out_prob)) / np.shape(out_true_prob)[0] * 100.
    voice_regions_percentage = (len(np.where(out_true_prob == 1)[0])) / np.shape(out_true_prob)[0] * 100.
    non_voice_regions_percentage = (len(np.where(out_true_prob == 0)[0])) / np.shape(out_true_prob)[0] * 100.

    print('Precision: %2f' % res[0])
    print('Recall: %2f' % res[1])
    print('Fscore: %2f' % res[2])
    print('Error: %2f' % cls_error)
    print('Singing voice frames percentage %2f' % voice_regions_percentage)
    print('Non-singing voice frames percentage %2f' % non_voice_regions_percentage)

    print('-- Saving Results --')
    np.save(os.path.join('results', exp_settings['split_name'], 'schl_model_results.npy'), out_prob)
    np.save(os.path.join('results', exp_settings['split_name'], 'vad_true_targets.npy'), out_true_prob)

    return None


if __name__ == "__main__":
    np.random.seed(218)
    torch.manual_seed(218)
    torch.cuda.manual_seed(218)

    # Training
    #perform_training()

    # Testing
    perform_testing()

# EOF
