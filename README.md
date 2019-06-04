# Singing Voice Activity Detection for Wagner Operas
## A Brief Introduction
The goal is to detect the activity of operatic singing voice. Towards that aim, a supervised (deep) model
is proposed and discussed. The proposed model consists of three "modules" denoted as the "Representation Learning",
the "Latent Feature Learning", and the "Classifier" module. The three modules are cascaded and optimized jointly.
The "Classifier" refers to a typical feed-forward-fully-connected neural network followed by a sigmoid function.
The "Latent Feature Learning" consists of gated recurrent neural networks (due to the strong labeling of the data),
skip-connections (for interpretability of the results), and some feed-forward-fully-connected neural networks
for computing a low-dimensional feature space that is given to the "Classifier".

## Dataset
The dataset contains three recordings of of "The Valkyrie" opera from Wagner. Each recording refers to a different
conductor. We denote each recording/performance by the conductor name. Consequently we have the following recordings:
**Barenboim-Kupfer**, **Haitink**, and **Karajan**. For each recording the annotations (i.e. the activity) of the operatic
singing voice is given by the corresponding libretti. However, for the Karajan opera there are also human annotations
that have minor time-adjustments of the operatic singing voice. We only use the **Karajan** opera with the corresponding
human annotations for testing, and we currently experiment with 2 dataset splits. 

The first split uses the the  **Barenboim-Kupfer** and **Haitink** recordings and their corresponding labels for training,
while the **Karajan** opera and the non-corrected labels for validation. The aforementioned split is denoted in the
code as **split_a**. The second split, denoted as **split_b**, uses the **Barenboim-Kupfer** recording for training,
the **Haitink** recording for validation, and the **Karajan** recording with the human annotations for testing.


## Supervised model
### Input to the Model
Mel-spectograms extracted using the short-time Fourier transform with a hop size of 512 samples and an analysis size of 2048 samples.
The number of Mel-bands is set to 250, following some pilot experimental procedure. A total of 3 seconds is used as a
sequence input to the rest of the modules, yielding a number of 134 time-frames in total per data-point used for training.
### Representation Learning
State of the art approaches:
1. Per-Channel Energy Normalization (PCEN)
2. 0-mean Convolutional Neural Networks (0-μ CNN)
3. Low-Rank PCEN (LR-PCEN): Serving as an extension of the PCEN
### Latent Feature Learning
A stack of bi-directional gated recurrent neural networks which output some information that is used to **mask** the
mel spectrogram provided by the **Representation Learning** module. The masked output can be used for interpreting
the results, hoping for a better "interpretability" of each model.
### Classifier
The **Classifier** module consists of a few layers for reducing the dimensionality of the masked data. The yielded
three-dimensional representation is given to a feed-forward-fully-connected neural network followed by a sigmoid function
for outputing the activity of the operatic singing voice **per frame**.
### Optimization
Binary cross entropy and back-propagation of errors using the Adam algorithm as a solver


### Results
1. Storage
⋅⋅ All the binary files from the results can be found in the following seafile link: https://seafile.idmt.fraunhofer.de/d/8724ff410d714d6fb437/ .
Currently the results are reported for the two above mentioned data splits, that are denoted as **split_a** and **split_b**.
The same names are used for the code repository. For the predicted, by each model, and true labels please download them
from the above link, they are stored under the "*.npy" files.
2. Reproducibility
⋅⋅ For reproducing the results, executing the files in the "/scripts/" folder will provide the reported
results. If only the test-phase of each experiment is needed, the results from the optimization can be found under
the above seafile link. The downloaded material can be placed in the corresponding "/results/" folder of this code
repository and the "perform_training()" function in each script should be commented and the "perform_testing()" function
should be used instead.
3. Elementary Statistics
⋅⋅ Just for the sake of checking on the validity of the results of each model and experiment, below you can find the percentage
of singing voice activity versus the percentage of the non-singing voice time-frames.
In the test set **~53.93%** of the time-frames contain operatic **singing voice**, whereas **~46.07%** contains **non-singing voice**
parts. From a first view, the dataset looks very balanced, and thus the following results are significant in
detecting singing voice versus non-singing voice.
4. Experimental Results

### **split_a**
| Metric         | 0μ-CNN    |Schl. Model| PCEN    | LR-PCEN  |  ORACLE  |
| ---------------|:---------:|:---------:|:-------:|:--------:|:--------:|
| Precision      | 0.888     | 0.938     | 0.891   | 0.903    |0.991     |
| Recall         | 0.885     | 0.838     | 0.905   | 0.912    |0.990     |
| F1-Score       | 0.887     | 0.885     | 0.898   | 0.907    |0.990     |
| Error-rate (%) | 11.705 %  | 11.271 %  | 10.610 %| 09.660 % |1.030%    |


### **split_a_ext**
| Metric         | 0μ-CNN    |Schl. Model| PCEN    | LR-PCEN  |  ORACLE  |
| ---------------|:---------:|:---------:|:-------:|:--------:|:--------:|
| Precision      | --        | 0.954     |  0.915  | 0.927    |0.991     |
| Recall         | --        | 0.964     |  0.949  | 0.934    |0.990     |
| F1-Score       | --        | 0.959     |  0.931  | 0.931    |0.990     |
| Error-rate (%) | --        | 4.237 %   |  7.203 %| 7.162 %  |1.030%    |



### **split_b**
| Metric         | 0μ-CNN    |Schl. Model| PCEN    | LR-PCEN  |  ORACLE  |
| ---------------|:---------:|:---------:|:-------:|:--------:|:--------:|
| Precision      | 0.840     | 0.947     | 0.846   | 0.882    |0.991     |
| Recall         | 0.858     | 0.704     | 0.774   | 0.768    |0.990     |
| F1-Score       | 0.849     | 0.807     | 0.808   | 0.821    |0.990     |
| Error-rate (%) | 15.839 %  | 17.389 %  | 19.007 %| 17.354 % |1.030%    |


### **ofit**
| Metric         | 0μ-CNN    |Schl. Model| PCEN    | LR-PCEN  |  ORACLE  |
| ---------------|:---------:|:---------:|:-------:|:--------:|:--------:|
| Precision      | --        | 0.975     | 0.901   | 0.910    |0.991     |
| Recall         | --        | 0.962     | 0.915   | 0.940    |0.990     |
| F1-Score       | --        | 0.968     | 0.908   | 0.925    |0.990     |
| Error-rate (%) | --        | 3.236 %   | 9.542 % | 7.905 %  |1.030%    |


