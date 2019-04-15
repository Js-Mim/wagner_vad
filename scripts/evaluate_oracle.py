# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'Fraunhofer IDMT'

"""
    Oracle evaluation. Using the labels from the stamps of the libretti
    and compared against the manually annotated ones. Providing upper bound
    error.
"""

# imports
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support as prf
from tools import helpers
from tools.experiment_settings import exp_settings


def perform_testing():
    print('--- Performing Oracle Evaluation ---')
    testing_data_dict = helpers.csv_to_dict(training=False)
    keys = list(testing_data_dict.keys())
    testing_key = keys[0]  # Validate on the second composer
    print('Testing on: ' + ' '.join(testing_key))
    # Get data
    _, y_annotated, _ = helpers.fetch_data(testing_data_dict, testing_key)

    # Get data dictionary
    training_data_dict = helpers.csv_to_dict(training=True)
    training_keys = sorted(list(training_data_dict.keys()))[2]
    print('Using predictions from: ' + " ".join(training_keys))
    # Get data
    _, y_noisy, _ = helpers.fetch_data(training_data_dict, training_keys)

    res = prf(y_annotated, y_noisy, average='binary')
    cls_error = np.sum(np.abs(y_annotated - y_noisy))/np.shape(y_annotated)[0] * 100.

    print('Precision: %2f' % res[0])
    print('Recall: %2f' % res[1])
    print('Fscore: %2f' % res[2])
    print('Error: %2f' % cls_error)

    return None


if __name__ == "__main__":
    np.random.seed(218)

    # Testing
    perform_testing()


# EOF
