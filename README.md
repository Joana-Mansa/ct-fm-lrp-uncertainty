# CT Foundation Model, Layer-wise Relevance Propagation and Uncertainty

Fine-tuning a self-supervised CT foundation model on lung nodule malignancy,
explaining it with layer-wise relevance propagation instead of Grad-CAM, and
testing whether the explanations survive contact with a faithfulness metric.

**The short version: the pretraining helps, and none of the four attribution
methods beat a random baseline on deletion. The second result is the more useful
one, and section 4.3 explains why.**

---

## 1. Goal

Three questions, in order.

1. **Does self-supervised CT pretraining actually help?** "Foundation model" is
   worth claiming only if the pretrained weights beat the same architecture
   trained from scratch, and the place it should show most is where labels are
   scarce.
2. **Are the explanations faithful?** Attribution maps are claims about what the
   model used. The claim is testable: remove the voxels a map ranks highest and
   the prediction should collapse faster than if you remove random ones. Most
   papers publish the map and skip the test.
3. **Do explanations degrade where the model is uncertain?** If faithfulness
   falls as predictive entropy rises, then an explanation shown to a clinician is
   least trustworthy exactly where they most need it.

---

## 2. Data and model

### NoduleMNIST3D

| Property | Value |
|---|---|
| Source | [MedMNIST v2](https://medmnist.com/), derived from LIDC-IDRI, CC BY 4.0 |
| Content | 3D chest CT patches centred on lung nodules |
| Resolution | 64 x 64 x 64 |
| Task | benign against malignant |
| Train / val / test | 1,158 / 165 / 310 |
| Train balance | 863 benign, 295 malignant |

The classes are imbalanced roughly 3:1, so balanced accuracy and AUC are
reported alongside plain accuracy, loss is class-weighted, and model selection
uses validation AUC.

The 64^3 release is used rather than 28^3 because CT-FM downsamples by 16 and
a 28^3 volume is not divisible by it.

### CT-FM

| Property | Value |
|---|---|
| Checkpoint | [`project-lighter/ct_fm_feature_extractor`](https://huggingface.co/project-lighter/ct_fm_feature_extractor) |
| Architecture | SegResNet, 87.2M parameters |
| Pretraining | contrastive self-supervised learning on **148,000 CT scans** from the Imaging Data Commons |
| Reference | [arXiv:2501.09001](https://arxiv.org/abs/2501.09001) |

The encoder returns a five-level pyramid. We take the bottleneck, 512 channels
at 4^3, pool it, normalise, and attach a linear head.

### Three things that cost time and would have produced meaningless results

Recorded because each one fails quietly rather than loudly.

**The checkpoint does not load into MONAI's `SegResNetDS`.** The parameter names
differ. `load_state_dict(strict=False)` matches **0 of 161 tensors** and returns
without complaint, so training proceeds on a randomly initialised network while
the code looks correct and the README claims a foundation model. The
`lighter_zoo` loader is the one that works.

**The from-scratch arm must be the same architecture.** Constructing a second
SegResNet from a guessed config gave 19.9M parameters against the real 87.2M.
That ablation would have measured architecture size and reported it as an effect
of pretraining. The scratch arm here is the loaded model with every parameter
reinitialised.

**Pooling the wrong layer gives the head two features.** The full SegResNet
output is a 2-channel volume, so pooling it hands the linear layer two numbers.
Combined with unnormalised features, whose standard deviation is about 0.03,
training loss sat at 6.37 with AUC 0.50. Using the bottleneck and a LayerNorm
moved it to 0.51 and 0.78.

---

## 3. Method

**Ablation.** Every configuration trained twice, from CT-FM weights and from
random initialisation, at 10%, 25% and 100% of the training labels. Subsets are
stratified and drawn with a fixed seed so both arms see identical volumes. The
encoder trains at 1e-5 and the head at 1e-3, because fine-tuning 87M pretrained
parameters at the head's rate walks the representation away from what the
self-supervised training produced, which is the thing being measured.

**Attribution.** Four methods. Three LRP composites from
[zennit](https://github.com/chr5tphr/zennit), whose layer-type registry covers
`Conv3d` and the 3D pooling layers so it runs on a volumetric network without
modification, plus Grad-CAM as the baseline most medical imaging papers use.

| Method | Rule |
|---|---|
| `EpsilonPlusFlat` | positive contributions in the conv stack, flat at the input |
| `EpsilonGammaBox` | gamma rule, box rule at the input, suited to bounded inputs |
| `EpsilonAlpha2Beta1` | positive and negative contributions at a fixed 2:1 ratio |
| Grad-CAM | gradient-weighted bottleneck activations, upsampled |

Reporting one composite and calling it "the LRP explanation" hides how much the
choice of rule matters, so all three are shown.

**Faithfulness.** Deletion curves, insertion curves, and AOPC, each against a
random-order baseline. The absolute numbers depend on the model and the data, so
only the gap against random is interpretable.

**Uncertainty.** MC dropout, 20 passes, giving predictive entropy and mutual
information, plus expected calibration error.

---

## 4. Results

### 4.1 Self-supervised pretraining helps

![ablation](docs/figures/ablation.png)

Test AUC, single seed:

| Labels | n | CT-FM | scratch | gain |
|---|---|---|---|---|
| 10% | 116 | 0.8283 | 0.8168 | +0.0116 |
| 25% | 290 | 0.8472 | 0.8124 | +0.0348 |
| 100% | 1,158 | **0.8951** | 0.8733 | +0.0218 |

CT-FM initialisation wins at every label budget. At full labels it reaches
0.895 AUC with balanced accuracy 0.830, against 0.873 and 0.788 from scratch.

![training curves](docs/figures/training_curves.png)

**Caveat, stated plainly.** These are single runs. A second seed reversed the
10% result (scratch 0.805 against CT-FM 0.802), which means the smallest gain in
the table is inside seed noise. The 25% and 100% gains are larger and look more
robust, but three seeds would be needed to claim any of this properly. The
replicate is what exposed it, and running one is cheap.

### 4.2 Calibration

![calibration](docs/figures/calibration.png)

Expected calibration error **0.145** under MC dropout, with mean predictive
entropy 0.085. The model is confident and worse than its confidence suggests,
which is the usual direction for a small fine-tuned network on an imbalanced
task.

### 4.3 No attribution method beats random deletion

This is the result that matters most, and it is negative.

![faithfulness](docs/figures/faithfulness.png)

| Method | deletion AUC | insertion AUC | AOPC | AOPC over random | stability |
|---|---|---|---|---|---|
| random baseline | **0.587** | | 0.368 | | |
| LRP eps+flat | 0.819 | 0.972 | 0.147 | **-0.221** | 0.012 |
| LRP eps-gamma-box | 0.822 | 0.972 | 0.144 | **-0.224** | -0.015 |
| LRP alpha2-beta1 | 0.816 | 0.973 | 0.149 | **-0.219** | 0.220 |
| Grad-CAM | 0.712 | 0.974 | 0.249 | **-0.119** | 0.924 |

Lower deletion AUC means the prediction collapses faster, so a faithful map
should sit *below* the random baseline. All four sit well above it. Deleting the
voxels these methods rank highest damages the prediction **less** than deleting
random voxels.

**This is a property of the metric, not proof that the maps are worthless.**
Random deletion scatters replaced voxels uniformly through the volume, which at
5% already produces something no CT scanner would output. Attribution-guided
deletion removes a compact, contiguous region, which looks far more like a
plausible scan. The random baseline wins by pushing the input further out of
distribution, not by identifying evidence better. This failure mode of deletion
metrics is known and is exactly the kind of thing that gets skipped when a paper
reports only heatmaps.

Two things follow:

- **On this task and at this scale, deletion is not a usable faithfulness test.**
  A masking scheme that keeps inputs on the data manifold, or an insertion-only
  protocol, would be the correct next step.
- **Insertion tells a different story.** All four methods reach 0.972 to 0.974,
  meaning restoring the highest-ranked voxels onto a blurred volume recovers the
  prediction quickly and to a similar degree.

**Stability is where the methods genuinely separate.** Under input noise the
model itself ignores, Grad-CAM maps stay almost fixed (correlation 0.924) while
LRP maps essentially decorrelate (0.012, -0.015, 0.220). Grad-CAM is stable
because it is coarse: a 4^3 map upsampled to 64^3 cannot move much. LRP resolves
to input resolution and pays for it in sensitivity. Whether a fine map that moves
under imperceptible noise is more useful than a coarse one that does not is a
real question, and it is not settled by either metric here.

![attribution panel](docs/figures/attribution_panel.png)

### 4.4 Uncertainty against faithfulness

![uncertainty vs faithfulness](docs/figures/uncertainty_vs_faithfulness.png)

Spearman correlation between predictive entropy and per-sample AOPC:

| Method | rho |
|---|---|
| LRP eps+flat | +0.190 |
| LRP eps-gamma-box | +0.174 |
| LRP alpha2-beta1 | +0.221 |
| Grad-CAM | -0.124 |

The three LRP composites agree with each other and disagree with Grad-CAM. For
LRP the correlation is weakly **positive**, so explanations are, if anything,
slightly more faithful on the cases the model is least sure about. Grad-CAM
leans the other way.

All four correlations are weak, and with 64 volumes none is significant. The
honest reading is that this experiment is underpowered rather than that the
effect is absent. It is the right question and the sample size needs to be an
order of magnitude larger.

---

## 5. Reproducing

### Requirements

```
torch  monai  medmnist  zennit  lighter-zoo  numpy  scipy  matplotlib
```

One GPU. The full ablation is about 14 minutes on an A100.

On Volta cards such as the V100, pin `torch==2.5.1+cu121`. Torch 2.14 ships an
arch list starting at `sm_75` and every CUDA call fails with
`no kernel image is available for execution on the device`.

### Pipeline

```bash
# 1. Ablation: both arms at three label budgets (14 min on an A100)
PYTHONPATH=src python src/train.py --epochs 25 --batch-size 16 \
    --fractions 0.1 0.25 1.0

# 2. Seed replicates, for error bars
PYTHONPATH=src python src/train.py --epochs 25 --seed 1 --tag _seed1 \
    --fractions 0.1 0.25 1.0

# 3. Attribution, faithfulness and uncertainty
PYTHONPATH=src python src/analyse.py --n 64 --batch 4

# 4. Figures
PYTHONPATH=src python src/figures.py
```

---

## 6. Repository layout

```
src/model.py          CT-FM encoder with a classification head
src/train.py          fine-tuning and the label-efficiency ablation
src/explain.py        zennit LRP composites and the Grad-CAM baseline
src/faithfulness.py   deletion, insertion, AOPC and stability
src/uncertainty.py    MC dropout, ensembles, calibration
src/analyse.py        runs the explanation and uncertainty analysis
src/figures.py        every figure in this README
results/              metrics as JSON, per-sample arrays as npz
```

---

## 7. Limitations

**Single seed for the headline table.** Section 4.1 says what the one replicate
already showed: the 10% gain did not survive it. Three seeds minimum before any
of these numbers should be quoted.

**64 volumes in the explanation analysis.** Enough to see that the deletion
metric misbehaves, not enough to resolve a weak correlation between uncertainty
and faithfulness.

**Deletion baseline replaces voxels with the dataset mean.** A blurred or
inpainted baseline would stay closer to the data manifold and probably change the
conclusion in 4.3. This is the first thing to fix.

**Resolution.** 64^3 patches, not full chest CT. CT-FM was pretrained on whole
scans, so fine-tuning on small patches may not use the representation the way it
was designed to be used.

**The uncertainty work uses MC dropout only.** `uncertainty.py` implements deep
ensembles, but the members were not trained in time. Ensembles usually give
better-calibrated estimates than dropout and would strengthen section 4.4.

---

## 8. References

- Pai et al., *Vision Foundation Models for Computed Tomography*,
  [arXiv:2501.09001](https://arxiv.org/abs/2501.09001)
- Salahuddin et al., *Transparency of deep neural networks for medical image
  analysis: a review of interpretability methods*,
  [PubMed 34891095](https://pubmed.ncbi.nlm.nih.gov/34891095/)
- Salahuddin et al., *Counterfactuals and Uncertainty-Based Explainable Paradigm
  for the Automated Detection and Segmentation of Renal Cysts in CT*,
  [arXiv:2408.03789](https://arxiv.org/abs/2408.03789)
- Anders et al., *Software for Dataset-wide XAI* (zennit),
  [arXiv:2106.13200](https://arxiv.org/abs/2106.13200)
- Samek et al., *Evaluating the visualization of what a deep neural network has
  learned*, [arXiv:1509.06321](https://arxiv.org/abs/1509.06321)
