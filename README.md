# CT Foundation Models, Attribution and Uncertainty

This project studies whether **CT-FM, an existing model pretrained on CT scans, helps classify 3D lung-nodule patches when labels are limited**. It compares fine-tuning with training the same architecture from scratch, then examines attribution maps and prediction uncertainty.

**Main finding:** pretraining gives small, variable average gains in the recorded experiments. The attribution maps are useful to inspect, but their faithfulness has not been established.

[Attribution gallery](docs/attribution.md) · [Architecture and losses](docs/architecture.md) · [Full results](docs/results.md) · [Run it](docs/reproduce.md) · [Verification](docs/verification.md)

## Data and task

| Item | Description |
|---|---|
| Dataset | NoduleMNIST3D, derived from LIDC-IDRI through MedMNIST v2 |
| Input | A grayscale 64 × 64 × 64 CT patch centred on a lung nodule |
| Target | Two categories derived from radiologists' malignancy ratings, **not biopsy-confirmed diagnoses** |
| Train / validation / test | 1,158 / 165 / 310 patches |
| This project's contribution | Downstream fine-tuning, matched scratch controls, attribution and uncertainty experiments |

CT-FM pretraining belongs to **Pai et al.** This project adapts their encoder to the nodule task. [Data provenance and label definitions](docs/data.md).

## Attribution maps: Grad-CAM and Zennit LRP

An attribution map assigns values to parts of the input to help inspect a model's prediction. **Grad-CAM** uses gradients and deep feature maps; **Zennit** supplies the three layer-wise relevance propagation (LRP) composites compared here.

![Real CT inputs beside Grad-CAM and three Zennit LRP maps, with test IDs, labels and predictions](docs/figures/attribution_preview.png)

**Read left to right:** CT input, Grad-CAM, EpsilonPlusFlat, EpsilonGammaBox and EpsilonAlpha2Beta1. These are saved outputs for test cases **257 and 180**, the first two cases of the original analysis subset. Each row shows the same slice and explains the predicted class. Label 0 means lower malignancy ratings; label 1 means higher ratings.

**Colours are relative:** brighter means higher within that normalised map. Each map was scaled separately, so colours cannot compare absolute importance across methods or distinguish positive from negative evidence. These are exploratory explanations, not segmentation masks.

[Four-case gallery, a misclassified example and interpretation guide](docs/attribution.md) · [Saved maps](results/qualitative.npz) · [Verified case identities](results/attribution_examples.json)

<details>
<summary>Inspect additional real input volumes in three views</summary>

![Real test patches 0 and 7, displayed along three array axes](docs/figures/data_samples.png)

These are the bundled input examples, separate from the attribution cases above. [Download and inspect the sample volumes](examples/README.md).

</details>

## Model architecture

![CT-FM encoder stages, feature sizes and classification head](docs/figures/architecture.svg)

The encoder converts the patch into **512 learned features**. Normalisation, dropout and a two-class classification head turn these into a prediction. The complete model has **77,763,042 parameters**.

Both encoder and head are trained. The pretrained and scratch arms use the same architecture and label subset. Attribution and uncertainty analyses happen after training. [Layer-by-layer details](docs/architecture.md).

## Training objective and learning curves

| Setting | Implementation |
|---|---|
| Loss | Class-weighted cross-entropy: penalise wrong predictions, with more weight on the less common label |
| Optimiser | AdamW; encoder learning rate 1e-5, head learning rate 1e-3 |
| Schedule | 25 epochs with cosine learning-rate decay |
| Checkpoint selection | Highest validation ROC AUC |

**ROC AUC measures how well the model ranks the two classes across decision thresholds.** Cross-entropy trains the model; validation AUC selects the saved checkpoint.

![Seed-0 full-data training losses and validation AUC for pretrained and scratch models](docs/figures/learning_curves.svg)

These curves come from the **seed-0, full-data** logs. Dots mark the selected checkpoints: epoch 11 for CT-FM and epoch 6 for scratch. Training loss keeps falling while validation performance fluctuates. [Loss equation and curve interpretation](docs/architecture.md).

## Results and what they mean

Mean test ROC AUC over **three matched seeds in the completed experiment records**:

| Training labels | Patches | CT-FM | Scratch | Difference |
|---|---:|---:|---:|---:|
| 10% | 116 | 0.8087 | 0.7923 | +0.0164 |
| 25% | 290 | 0.8591 | 0.8455 | +0.0136 |
| 100% | 1,158 | 0.8955 | 0.8898 | +0.0058 |

**Classification:** re-evaluating the full-data seed-0 checkpoints gives **0.8951 AUC for CT-FM** and **0.8733 for scratch**. The 95% patch-bootstrap interval for their difference is **[-0.0184, 0.0659]**, which includes zero. Some individual runs favour scratch.

**Attribution:** on 64 patches, randomly masking voxels reduced confidence more than masking the voxels ranked highest by any of the four methods. That result does not validate the maps' faithfulness. Zennit rule handling and the masking procedure need further controls.

**Uncertainty:** repeated predictions with dropout in the classification head were compared with attribution behaviour. The 64-case analysis did not establish a clear relationship between uncertainty and the perturbation-based explanation scores.

Lower-budget scores and full-data scratch seed 1 remain **record-based** because those checkpoints were unavailable for re-evaluation. [All individual runs, comparison plots and calibration results](docs/results.md) · [Record reconciliation and verification scope](docs/verification.md).

## Try it

Inspect the bundled inputs and render the saved attribution maps on a CPU:

```bash
git clone https://github.com/Joana-Mansa/ct-fm-lrp-uncertainty.git
cd ct-fm-lrp-uncertainty
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
python scripts/preview_data.py
python scripts/attribution_figures.py
```

These commands need no model or full-dataset download. For new predictions, use the [reproduction guide](docs/reproduce.md), which includes checked seed-0 weights and checkpoint verification.

## Scope and next validation needs

- This is a preprocessed patch benchmark; external cohorts, whole-CT diagnosis and clinical evaluation are not covered.
- The distributed arrays lack patient/site identifiers. Patient-level split independence and overlap with CT-FM pretraining data were not independently audited.
- The attribution and head-dropout uncertainty outputs remain exploratory. Anatomical plausibility, relevance conservation and clinical reliability need further validation.

[Technical report](paper/ctfm_lrp_uncertainty.pdf) (not peer reviewed) · [Detailed methods](docs/methods.md) · [Source code](src/)

**Credits:** [MedMNIST v2](https://medmnist.com/), CC BY 4.0; [Pai et al., CT-FM](https://arxiv.org/abs/2501.09001), [official weights](https://huggingface.co/project-lighter/ct_fm_feature_extractor), Apache 2.0; [Zennit](https://github.com/chr5tphr/zennit).
