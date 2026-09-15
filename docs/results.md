[Overview](../README.md) · [Data](data.md) · [Methods](methods.md) · [Results](results.md) · [Run it](reproduce.md) · [Verification](verification.md)

# Results

## Matched label-efficiency runs

Only seed/budget pairs with both pretrained and scratch records are averaged. Unpaired full-data seed 1 and seed 10 pretrained models are excluded from the paired summary. Lower-budget weights were not retained, so those scores are checked against saved records rather than rerun from checkpoints.

| Seed | Labels | Patches | CT-FM AUC | Scratch AUC | Difference |
|---|---:|---:|---:|---:|---:|
| 0 | 10% | 116 | 0.8283 | 0.8168 | +0.0116 |
| 0 | 25% | 290 | 0.8472 | 0.8124 | +0.0348 |
| 0 | 100% | 1158 | 0.8951 | 0.8733 | +0.0218 |
| 1 | 10% | 116 | 0.8143 | 0.8053 | +0.0090 |
| 1 | 25% | 290 | 0.9017 | 0.8711 | +0.0306 |
| 2 | 10% | 116 | 0.7936 | 0.7548 | +0.0388 |
| 2 | 25% | 290 | 0.8552 | 0.8533 | +0.0018 |
| 2 | 100% | 1158 | 0.8805 | 0.8860 | -0.0055 |

Mean paired gains are +0.0198 (10%, three seeds), +0.0224 (25%, three seeds), and +0.0081 (100%, two seeds). All recorded 10% pairs favour CT-FM; the full-data seed-2 pair favours scratch. This does not establish universal superiority or a monotonic label-efficiency effect.

## Full-data checkpoint verification

All six available checkpoints were strictly loaded and evaluated on the same 310 test patches (246 label 0, 64 label 1).

| Checkpoint | ROC AUC | Accuracy | Balanced accuracy |
|---|---:|---:|---:|
| `ctfm_pretrained.pt` | 0.8951 | 0.8226 | 0.8304 |
| `ctfm_pretrained_ens10.pt` | 0.9093 | 0.8774 | 0.8303 |
| `ctfm_pretrained_seed1.pt` | 0.9111 | 0.8516 | 0.7678 |
| `ctfm_pretrained_seed2.pt` | 0.8805 | 0.5839 | 0.6916 |
| `ctfm_scratch.pt` | 0.8733 | 0.7548 | 0.7877 |
| `ctfm_scratch_seed2.pt` | 0.8860 | 0.8129 | 0.7839 |

For the seed-0 paired AUC difference (+0.0218), a 2,000-replicate stratified bootstrap (seed 2026) gives a 95% percentile interval **[-0.0184, 0.0659]**. Resampling unit is the benchmark patch, not the patient. This interval does not capture training-seed variability or unknown patient dependence.

AUC measures ranking, while accuracy uses a decision threshold. The pretrained seed-2 accuracy of 0.5839 despite AUC 0.8805 shows why both must be reported. The full-data seed-1 pretrained checkpoint exists, but its corresponding full training record and scratch counterpart are absent from the saved ablation JSON.

## Attribution and perturbation metrics

The original analysis uses 64 test patches; stability uses only its first 16. Random masking is a comparator, not an attribution method. Lower deletion AUC and higher AOPC indicate a larger confidence drop under this particular masking procedure.

| Method | Deletion AUC | Insertion AUC | AOPC | AOPC minus random | Stability (n=16) |
|---|---:|---:|---:|---:|---:|
| Random | 0.5870 | n/a | 0.3679 | 0 | n/a |
| EpsilonPlusFlat | 0.8193 | 0.9718 | 0.1466 | -0.2213 | 0.0124 |
| EpsilonGammaBox | 0.8224 | 0.9719 | 0.1436 | -0.2242 | -0.0146 |
| EpsilonAlpha2Beta1 | 0.8164 | 0.9733 | 0.1494 | -0.2185 | 0.2201 |
| Grad-CAM | 0.7121 | 0.9736 | 0.2487 | -0.1192 | 0.9235 |

![Perturbation results with random-order comparison](figures/faithfulness.png)

All four methods produce a smaller confidence drop than random deletion. This does not validate their faithfulness. Neither a general failure of LRP nor a causal explanation based on distribution shift has been established: the current rule handling and masking baseline both need controls. Grad-CAM's stable, coarse maps are not independently proven faithful.

## Uncertainty and calibration

| Analysis | Number of test patches | AUC | ECE (10 bins) |
|---|---:|---:|---:|
| Original seed-0 MC-dropout explanation subset | 64 | Not reported here | 0.1453 |
| Seed-0 MC dropout, 20 passes, full test set | 310 | 0.8960 | 0.1265 |
| Four-checkpoint probability ensemble | 310 | 0.9085 | 0.0505 |

The ensemble's AUC is below the best individual member (0.9111). Its lower observed ECE on this test set is promising but not a deployment or external-calibration claim. The deterministic ensemble metrics were recomputed from all four checkpoints. The stochastic MC-dropout outputs were checked from saved arrays, not exactly regenerated.

Entropy versus AOPC, recomputed with SciPy from the saved 64-case arrays:

| Attribution | Spearman rho | Two-sided p-value |
|---|---:|---:|
| EpsilonPlusFlat | 0.1904 | 0.1318 |
| EpsilonGammaBox | 0.1739 | 0.1692 |
| EpsilonAlpha2Beta1 | 0.2214 | 0.0787 |
| Grad-CAM | -0.1245 | 0.3270 |

All p-values exceed 0.05 before any multiple-comparison adjustment. No clear uncertainty/faithfulness relationship is established. Head-only dropout cannot establish whether uncertainty is intrinsically aleatoric or whether an explanation is reliable.

## Inspect a misclassified case

![A real nodule patch, exploratory attribution maps and dropout probabilities](figures/inference_example.png)

The documented command uses test index 0 and seed 2026. Its rating-derived label is `benign`, but the model predicts `malignant` with probability 0.645. The displayed heatmaps do not make that prediction correct. MC-dropout entropy is 0.657 across 20 passes. This first-index example is retained to show a real failure rather than select only correct predictions. [Exact output and label qualification](../results/inference_example.json).

## Evidence files

[Seed 0](../results/ablation.json) · [Seed 1](../results/ablation_seed1.json) · [Seed 2](../results/ablation_seed2.json) · [Additional ensemble member](../results/ablation_ens10.json)

[Attribution summary](../results/explanations.json) · [64-case arrays](../results/per_sample.npz) · [Ensemble summary](../results/ensemble.json) · [Ensemble arrays](../results/ensemble_per_sample.npz)

[Verification JSON](../results/verification.json) · [All six checkpoint predictions](../results/checkpoint_predictions.npz) · [Verification scope](verification.md)
