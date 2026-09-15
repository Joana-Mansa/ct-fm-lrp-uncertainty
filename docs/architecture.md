[Overview](../README.md) · [Methods](methods.md) · [Results](results.md) · [Run it](reproduce.md)

# Architecture and training objectives

![CT-FM encoder and classification head](figures/architecture.svg)

## From a nodule patch to a prediction

Shapes are `channels × depth × height × width`; batch size is omitted. Each residual block uses two 3³ convolutions with **BatchNorm3d and ReLU**, plus a shortcut. The initial stem maps one input channel to 32 channels. Stride-2 convolutions between stages halve the spatial dimensions.

| Encoder stage | Residual blocks | Feature shape |
|---|---:|---|
| 0 | 1 | 32 × 64 × 64 × 64 |
| 1 | 2 | 64 × 32 × 32 × 32 |
| 2 | 2 | 128 × 16 × 16 × 16 |
| 3 | 4 | 256 × 8 × 8 × 8 |
| 4 | 4 | 512 × 4 × 4 × 4 |

The classifier uses only the deepest feature map. Global average pooling reduces it to a 512-element vector, then LayerNorm, dropout with probability 0.3, and a linear layer produce two logits. Softmax converts logits into probabilities. The original CT-FM decoder is discarded; this is a classification model, not a segmentation network.

Both the encoder and head are optimised during fine-tuning. The scratch arm uses the same architecture with reinitialised encoder weights. This is not a frozen-encoder linear-probe experiment. The total classifier has **77,763,042 parameters**. [Model source](../src/model.py) · [Measured stage shapes and parameter counts](../results/architecture.json).

## Class-weighted cross-entropy

The training set contains 863 lower-rating and 295 higher-rating patches. For a training subset of size $N$ with $n_c$ examples in class $c$, the script sets:

$$
w_c=\frac{N}{2n_c}.
$$

For the full training set, the weights are approximately **0.671 for label 0** and **1.963 for label 1**. They are recalculated for each smaller stratified subset. The less common label therefore receives a larger contribution to the loss.

For a mini-batch $\mathcal{B}$, target $y_i$ and softmax probability $p_i$, PyTorch's weighted mean reduction computes:

$$
\mathcal{L}_{\mathrm{nodule}}=
-\frac{\sum_{i\in\mathcal{B}}w_{y_i}\log p_i(y_i)}{\sum_{i\in\mathcal{B}}w_{y_i}}.
$$

The code passes raw logits to cross-entropy. Class weighting changes the training penalty; it does not balance the test set or guarantee calibrated probabilities. Labels remain radiologist-rating-derived categories, not biopsy-confirmed diagnoses.

| Training setting | Value |
|---|---|
| Optimiser | AdamW, weight decay 0.0001 |
| Encoder learning rate | 0.00001 |
| Classification-head learning rate | 0.001 |
| Schedule | Cosine learning-rate decay over 25 epochs |
| Batch size | 8 in the training script |
| Label budgets | 116, 290 and 1,158 patches, shared between matched arms |
| Checkpoint selection | Highest validation ROC AUC |

AUC measures how well scores rank the two classes across thresholds. It is used to **select** a checkpoint; cross-entropy is the differentiable loss used to **train** it. [Training implementation](../src/train.py).

## What the recorded curves show

![Full-data seed-0 training loss and validation ROC AUC](figures/learning_curves.svg)

These plots use only the two full-data runs in [`results/ablation.json`](../results/ablation.json), seed 0. They are not multi-seed averages. Dots mark the first maximum validation AUC, matching the script's checkpoint-selection rule:

| Arm | Selected epoch | Validation AUC | Re-evaluated test AUC |
|---|---:|---:|---:|
| CT-FM fine-tuning | 11 | 0.8815 | 0.8951 |
| Scratch | 6 | 0.8804 | 0.8733 |

Training loss falls toward zero while validation AUC fluctuates. The lowest training-loss checkpoint is therefore not necessarily the selected checkpoint. No validation cross-entropy was logged, so no validation-loss curve is shown. The logged training curve averages each batch's weighted-mean loss using batch size; it is not a fresh whole-dataset weighted-loss calculation.

The [complete results](results.md) report all recorded seeds and the uncertainty around the test-set comparison. These curves alone do not establish a reliable pretraining advantage.

## Analysis after training

| Analysis | Where it acts | What changes |
|---|---|---|
| Grad-CAM | Deepest 512 × 4³ feature map, upsampled to 64³ | Computes an attribution map; no weight updates |
| Zennit composites | Backward attribution through the classifier | Exploratory rules; architecture-specific conservation remains unvalidated |
| MC dropout | Dropout in the classification head | Repeated stochastic predictions with fixed weights |
| Deletion / insertion | Input voxels | Measures model response to perturbed inputs |

There is **no attribution, uncertainty or faithfulness term in the training loss**. The project evaluates these properties afterward. In particular, finite Zennit maps do not establish that the residual and normalisation operations have validated relevance rules.

Regenerate the SVG/PNG diagrams and curves with `python scripts/architecture_figures.py`. The feature shapes were checked with a forward pass through the actual instantiated encoder. An earlier methods description incorrectly named GroupNorm; the runtime encoder uses BatchNorm3d, and the text is corrected.
