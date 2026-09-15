[Overview](../README.md) · [Data](data.md) · [Methods](methods.md) · [Results](results.md) · [Run it](reproduce.md) · [Verification](verification.md)

# Method

## Model and matched controls

The official `project-lighter/ct_fm_feature_extractor` checkpoint is loaded through `lighter_zoo.SegResNet.from_pretrained`. This uses the pretrained encoder's deepest 512-channel, 4 × 4 × 4 feature map, followed by global average pooling, LayerNorm, dropout 0.3 and a two-class linear head. The resulting classifier has **77,763,042 parameters**, confirmed by loading the saved weights.

The scratch arm reinitialises the same architecture. It is not a smaller independently configured network. CT-FM pretraining was carried out by [Pai et al.](https://arxiv.org/abs/2501.09001); this project performs downstream fine-tuning and evaluation.

At each label budget, both arms use the same stratified subset for a given seed: 116, 290 or 1,158 training patches. Training uses class-weighted cross-entropy, AdamW, encoder learning rate 1e-5, head learning rate 1e-3, cosine decay and 25 epochs. Best validation AUC selects the checkpoint. Test AUC, accuracy and balanced accuracy are reported. The NPZ preprocessing is a benchmark adaptation, not the original full-scan CT-FM preprocessing.

## Attribution

Three Zennit composites (`EpsilonPlusFlat`, `EpsilonGammaBox`, `EpsilonAlpha2Beta1`) and bottleneck Grad-CAM are applied to the predicted class. Grad-CAM is computed on a 4³ map and interpolated to 64³, so smoothness is expected.

**Implementation status:** the Zennit-composite maps are exploratory. This architecture contains residual additions, GroupNorm and LayerNorm, and no architecture-specific canonizer or full relevance-conservation validation is supplied. Two-case checks confirm finite maps of the expected shape, but their sums are not evidence of conservation. Consequently these results do not establish a general failure or success of LRP. See the [Zennit canonizer documentation](https://zennit.readthedocs.io/en/stable/how-to/use-rules-composites-and-canonizers.html).

## Deletion, insertion and stability

- **Deletion:** progressively replace top-ranked voxels with `x.mean().item()`, the mean of the current input batch. It is not a fixed dataset mean. Compare target-class probabilities with a random voxel ordering.
- **Insertion:** progressively restore ranked voxels into a 5 × 5 × 5 average-blurred input.
- **Deletion/insertion AUC:** area under these probability curves. These are perturbation metrics, distinct from classification ROC AUC.
- **AOPC:** average drop from the original target probability during deletion. `AOPC over random` subtracts the random-order baseline.
- **Stability:** correlation of clean and noisy maps, Gaussian noise sigma 0.05, three repeats, on the **first 16** of the 64 explanation cases.

Masking may produce inputs outside the training distribution. That is a possible explanation for the random baseline's strong effect, not a tested causal conclusion. Alternative baselines and rule-validation controls are needed before drawing broader conclusions.

## Uncertainty and calibration

MC dropout uses 20 stochastic passes with dropout in the **classification head**. Predictive entropy measures uncertainty in mean probabilities; mutual information measures variation across these passes. This is a limited approximation, not a full model posterior or a clinical reliability guarantee.

ECE uses 10 equal-width confidence bins. Keep sample sizes separate: the original explanation analysis uses 64 patches, while ensemble comparisons use all 310. Ensemble predictions average probabilities from four pretrained checkpoints (seeds 0, 1, 2 and 10).

Spearman correlations now use SciPy's tie-aware implementation. Recalculation from the saved arrays leaves the reported coefficients unchanged and adds unadjusted two-sided p-values.
