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
Mel-spectograms extracted using ...
### Representation Learning
State of the art approaches:
1. Per-Channel Energy Normalization (PCEN)
2. 0-mean Convolutional Neural Networks (0-μ CNN)
3. Low-Rank PCEN (LR-PCEN): Serving as an extension of the PCEN
### Latent Feature Learning
### Classifier
Feed-forward-fully-connected neural network followed by a sigmoid function

