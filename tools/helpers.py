# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

# imports
import os
import csv
import re
import numpy as np
from tools import io_methods as io
from tools.experiment_settings import exp_settings

# definitions
path_to_wagner = '/mnt/IDMT-WORKSPACE/DATA-STORE/mis/Datasets/WagnerLyrics/'
path_to_wagner_full_ring = '/mnt/IDMT-WORKSPACE/DATA-STORE/mis/Datasets/WagnerLyrics-FullRing/'
annotated_sheet = 'Annotations_SingingVoice_SheetMusic'
human_annotations = 'Annotations_SingingVoice_Audio'
wav_files = 'wav_22050_stereo'


def csv_to_dict(training=True):
    """
        A function to return a dictionary using the information of the ".csv" files
        of the Wagner dataset.
        Arguments:
            training       :    (bool)   Grab training or testing data
        Returns:
            data_dict      :    (dict)   Dictionary of the following (nested) items:
                                         'recording'          : Name of the recording
                                             - 'singer'       : Name of the singer
                                             - 'lyrics_list'  : Lyrics
                                             - 'start_time'   : Starting time stamp
                                             - 'stop_time'    : Ending time stamp
                                             - 'wav_path'     : The path to the wav file
    """
    if training:
        path_to_csv = os.path.join(path_to_wagner, annotated_sheet)
    else:
        path_to_csv = os.path.join(path_to_wagner, human_annotations)

    wav_files_list = os.listdir(os.path.join(path_to_wagner, wav_files))

    csv_files = os.listdir(path_to_csv)

    # Punctuation
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~-'''

    # Creating dictionary keys using the conductors
    csv_index = 0
    for csv_element in csv_files:
        csv_files[csv_index] = re.match(r"([a-z]+)([0-9]+)", csv_element.split('_')[2], re.I).groups()[0]
        csv_index += 1
    data_dict = dict.fromkeys(sorted(csv_files), {})

    # Training/Validation set
    for item in os.listdir(path_to_csv):
        if item.endswith('.csv'):
            # Find out the exact wav file path based on the conductors
            path_to_wav = []
            for wav_file in wav_files_list:
                stripped_file_name = re.match(r"([a-z]+)([0-9]+)", wav_file.split('_')[2], re.I).groups()[0]

                if stripped_file_name == re.match(r"([a-z]+)([0-9]+)", item.split('_')[2], re.I).groups()[0]:
                    path_to_wav = os.path.join(path_to_wagner, wav_files, wav_file)

            if 'path_to_wav' not in locals():
                path_to_wav = []
                print('Audio file missing for annotation: ' + item)

            with open(os.path.join(path_to_csv, item), 'r', encoding='ISO-8859-1') as csvfile:
                csvreader = csv.reader(csvfile, delimiter='/')
                next(csvreader, None)
                # Lists
                singer_list = []
                start_time_list = []
                stop_time_list = []
                lyrics_list = []
                for row in csvreader:
                    singer_list += [row[1]]
                    start_time_list += [float(row[2])]
                    stop_time_list += [float(row[3])]

                    # Grab the lyrics
                    current_lyrics = row[4].split()
                    word_indx = 0
                    for word in current_lyrics:
                        np_word = ""
                        for char in word:
                            if char not in punctuations:
                                np_word += char
                        current_lyrics[word_indx] = np_word
                        word_indx += 1

                    lyrics_list += [current_lyrics]

                data_dict[re.match(r"([a-z]+)([0-9]+)",
                                   item.split('_')[2], re.I).groups()[0]] = {'singer': singer_list,
                                                                             'lyrics': lyrics_list,
                                                                             'start_time': start_time_list,
                                                                             'stop_time': stop_time_list,
                                                                             'wav_path': path_to_wav}

    return data_dict


def fetch_data(data_annotations, key):
    """
        A function to return data used for training
        given a dictionary of a dataset.
        Args:
            data_annotations  :  (dict)  Dictionary of the following (nested) items:
                                             - 'recording'    : Name of the recording
                                             - 'singer'       : Name of the singer
                                             - 'start_time'   : Starting time stamp
                                             - 'stop_time'    : Ending time stamp
                                             - 'wav_path'     : The path to the wav file
            key               :  (str)   String for accessing a composer, or a list of strings
                                         for obtaining the data from multiple composers.
    """
    def _fetch_data(data, a_key):
        # Get the path and read the waveform
        wav_file_path = data[a_key]['wav_path']
        out_x, out_fs = io.AudioIO.wavRead(wav_file_path, mono=True)
        # Generate time-domain labels
        pointers_in = data[a_key]['start_time']
        pointers_out = data[a_key]['stop_time']
        if not len(pointers_in) == len(pointers_out):
            raise AttributeError("Unequal number of pointers. Problems may occur...")
        out_y = np.zeros(out_x.shape)
        for p_indx in range(len(pointers_in)):
            c_pin = int(np.floor(pointers_in[p_indx] * out_fs))
            c_pout = int(np.floor(pointers_out[p_indx] * out_fs))
            out_y[c_pin:c_pout] = 1.

        return out_x, out_y, out_fs

    if type(key) == list:
        print('Number of key entries: ' + str(len(key)))
        print('Fetching: ' + key[0])
        x, y, fs = _fetch_data(data_annotations, key[0])
        for key_item in key[1:]:
            print('Fetching: ' + key_item)
            x_b, y_b, _ = _fetch_data(data_annotations, key_item)
            x = np.hstack((x, x_b))
            y = np.hstack((y, y_b))
    else:
        x, y, fs = _fetch_data(data_annotations, key)

    return x, y, fs


def fetch_data_with_singer_ids(data_annotations, key):
    """
        A function to return data used for training
        given a dictionary of the Wagner dataset.
        Compared to the "fetch_data" function this returns
        the labels for multi-class scenario.
        Args:
            data_annotations  :  (dict)  Dictionary of the following (nested) items:
                                             - 'recording'    : Name of the recording
                                             - 'singer'       : Name of the singer
                                             - 'start_time'   : Starting time stamp
                                             - 'stop_time'    : Ending time stamp
                                             - 'wav_path'     : The path to the wav file
            key               :  (str)   String for accessing a composer, or a list of strings
                                         for obtaining the data from multiple composers.
    """
    def _fetch_data(data, a_key):
        # Get the path, the singer ids, and read the waveform
        wav_file_path = data[a_key]['wav_path']
        singer_id = data[a_key]['singer']
        singer_list = []
        for singer in singer_id:
            if singer not in singer_list:
                singer_list.append(singer)
        singer_list = sorted(singer_list)
        out_x, out_fs = io.AudioIO.wavRead(wav_file_path, mono=True)

        # Generate time-domain labels
        pointers_in = data[a_key]['start_time']
        pointers_out = data[a_key]['stop_time']
        if not len(pointers_in) == len(pointers_out):
            raise AttributeError("Unequal number of pointers. Problems may occur...")
        if not len(pointers_in) == len(singer_id):
            raise AttributeError("Unequal number of pointers and singer ids. Problems may occur...")

        out_y = np.zeros((out_x.shape[0], len(singer_list) + 1))

        for p_indx in range(len(pointers_in)):
            c_pin = int(np.floor(pointers_in[p_indx] * out_fs))
            c_pout = int(np.floor(pointers_out[p_indx] * out_fs))
            singer_class = [class_index for class_index, singer_element in
                            enumerate(singer_list) if singer_element == singer_id[p_indx]][0] + 1
            out_y[c_pin:c_pout, 0] = 1.
            out_y[c_pin:c_pout, singer_class] = 1.

        return out_x, out_y, out_fs, singer_list

    if type(key) == list:
        print('Number of key entries: ' + str(len(key)))
        print('Fetching: ' + key[0])
        x, y, fs, list_of_singers = _fetch_data(data_annotations, key[0])
        for key_item in key[1:]:
            print('Fetching: ' + key_item)
            x_b, y_b, _, _ = _fetch_data(data_annotations, key_item)
            x = np.hstack((x, x_b))
            y = np.vstack((y, y_b))
    else:
        x, y, fs, list_of_singers = _fetch_data(data_annotations, key)

    return x, y, fs, list_of_singers


def build_wagner_vocabulary(data_annotations, keys_list):
    """
        A function to build a vocabulary using the lyrics
        from Wagner operas.
        Args:
            data_annotations  :  (dict)  Dictionary of the following (nested) items:
                                         'recording'    : Name of the recording
                                             - 'singer'       : Name of the singer
                                             - 'start_time'   : Starting time stamp
                                             - 'stop_time'    : Ending time stamp
                                             - 'wav_path'     : The path to the wav file
            keys_list         :  (list)  List containing strings for each composer.
    """
    vocab = list('SOS')
    vocab.append('EOS')
    vocab.append('SLC')

    for key in keys_list:
        lyric_tokens = data_annotations[key]['lyrics']

        for sentence in lyric_tokens:
            for token in sentence:
                if token not in vocab:
                    vocab.append(token)

    word2indx = {w: indx for (indx, w) in enumerate(vocab)}
    indx2word = {indx: w for (indx, w) in enumerate(vocab)}

    return word2indx, indx2word


def gimme_batches(batch_indx, data_points, x_in, y_out):

    batch_data_points = data_points[batch_indx * exp_settings['batch_size']:
                                    (batch_indx + 1) * exp_settings['batch_size']]

    d_p_length_samples = exp_settings['d_p_length'] * exp_settings['fs']

    x_d_ps = np.zeros((exp_settings['batch_size'], d_p_length_samples))
    y_d_ps = np.zeros((exp_settings['batch_size'], d_p_length_samples))
    storing_index = 0

    for data_point in batch_data_points:
        # Chop data
        x_s = x_in[data_point * d_p_length_samples:(data_point + 1) * d_p_length_samples]\
            .reshape(1, d_p_length_samples)

        y_s = y_out[data_point * d_p_length_samples:(data_point + 1) * d_p_length_samples]\
            .reshape(1, d_p_length_samples)

        x_d_ps[storing_index, :] = x_s
        y_d_ps[storing_index, :] = y_s
        storing_index += 1

    return x_d_ps, y_d_ps


def gimme_batches_multi_class(batch_indx, data_points, y_clusters):

    batch_data_points = data_points[batch_indx * exp_settings['batch_size']:
                                    (batch_indx + 1) * exp_settings['batch_size']]

    d_p_length_samples = exp_settings['d_p_length'] * exp_settings['fs']

    num_of_clusters = y_clusters.shape[-1]

    y_d_ps = np.zeros((exp_settings['batch_size'], d_p_length_samples, num_of_clusters))
    storing_index = 0

    for data_point in batch_data_points:
        # Chop data
        y_s = y_clusters[data_point * d_p_length_samples:(data_point + 1) * d_p_length_samples]\
            .reshape(1, d_p_length_samples, num_of_clusters)

        y_d_ps[storing_index, :] = y_s
        storing_index += 1

    return y_d_ps


if __name__ == '__main__':
    training_data_dict = csv_to_dict(training=True)
    training_keys = list(training_data_dict.keys())
    testing_data_dict = csv_to_dict(training=False)
    testing_keys = list(testing_data_dict.keys())
    # Build vocabulary
    #word2indx, indx2word = build_wagner_vocabulary(data_dict, keys)
    # Get training/validation data
    x, y, fs = fetch_data(training_data_dict, training_keys[0])
    # Get training/validation data with singer ids
    x, y, fs, singer_ids = fetch_data_with_singer_ids(training_data_dict, training_keys[0])


# EOF
