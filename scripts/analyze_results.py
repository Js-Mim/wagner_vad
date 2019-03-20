# -*- coding: utf-8 -*-
__author__ = 'Christof Weiss'
__copyright__ = 'AudioLabs Erlangen'

# imports
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import colors
from tools import helpers

# path_results = os.path.join('results','split_a')
path_results = os.path.join('results','split_b')

# filename = 'pcen_results.npy'
# filename = 'lr_pcen_results.npy'
filename = '0m_cnn_results.npy'

filenameTarget = 'vad_true_targets.npy'

results = np.load(os.path.join(path_results,filename))
targets = np.load(os.path.join(path_results,filenameTarget))

results = results.reshape((results.shape[0],1))
targets = targets.reshape((results.shape[0],1))

results = np.transpose(results)
targets = np.transpose(targets)

# results = np.transpose(results.reshape((results.shape[0],1)))
# targets = np.transpose(targets.reshape((results.shape[0],1)))

print(results.shape)
print(targets.shape)

truePositives = np.multiply(targets,results)
falsePositives = np.subtract(results,truePositives)
falseNegatives = np.subtract(targets,truePositives)

precision = np.sum(truePositives,axis=None)/(np.sum(truePositives,axis=None)+np.sum(falsePositives,axis=None))
recall = np.sum(truePositives,axis=None)/(np.sum(truePositives,axis=None)+np.sum(falseNegatives,axis=None))
fmeas = 2*precision*recall / (precision+recall)
error_rate = np.sum(falsePositives+falseNegatives,axis=None)/results.shape[1]

print('Precision:  '+str(precision))
print('Recall:     '+str(recall))
print('F-Measure:  '+str(fmeas))
print('Error rate: '+str(error_rate))

results_categories = 3*truePositives + 2*falseNegatives + 1*falsePositives

discreteCmap = colors.ListedColormap([[1,1,1], [1, 0.3, 0.3], [1, 0.7, 0.7], [0, 0, 0]])
bounds = [0,1,2,3,4]
norm = colors.BoundaryNorm(bounds, discreteCmap.N)

fig = plt.figure(figsize=(10, 4))
img = plt.imshow(results_categories, origin='lower', aspect='auto', norm=norm, interpolation='nearest', cmap=discreteCmap)
plt.clim([0, 4])
plt.xlabel('Time (frames)')
cbar = plt.colorbar(img, cmap=discreteCmap, norm=norm, boundaries=bounds, ticks=[0.5, 1.5, 2.5, 3.5])
cbar.ax.set_yticklabels(["TN","FP","FN","TP"])
plt.title('Results: WWV086B-1, Karajan, fine annotation')
plt.tight_layout()
plt.show()
