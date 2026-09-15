[Overview](../README.md) · [Methods](methods.md) · [Results](results.md) · [Run it](reproduce.md)

# Grad-CAM and Zennit attribution maps

The panel below shows **real saved outputs** from the seed-0 CT-FM classifier. Every row compares the same CT input and predicted class across Grad-CAM and three Zennit composites. These are existing experimental arrays, rendered with consistent display settings.

![Four real CT cases with Grad-CAM, EpsilonPlusFlat, EpsilonGammaBox and EpsilonAlpha2Beta1 maps](figures/attribution_panel.png)

## Which cases are shown?

These are the first four cases from the original 64-case analysis subset, selected with NumPy `default_rng(0)` without replacement from the 310 test patches. No cases were reordered or filtered by their appearance or prediction accuracy. Their saved volumes match the benchmark images exactly after preprocessing, and their predictions match the checkpoint audit.

| Test index | Rating-derived label | Prediction | Probability of predicted class |
|---|---|---|---:|
| 257 | 0: lower ratings | 0 | 0.9998 |
| 180 | 1: higher ratings | 1 | 0.9995 |
| 212 | 0: lower ratings | 0 | 0.9983 |
| 114 | 0: lower ratings | 0 | 0.9922 |

All four happen to be correctly classified. They do not estimate general performance. Probabilities come from the audited deterministic checkpoint predictions, not from MC-dropout averages.

## How to read the colours

- The input is a 64³ CT-derived patch scaled to [-1, 1]. Each column displays **array axis 0, slice 32**. The archive does not establish anatomical orientation.
- The saved attribution arrays were min-max normalised over each full 3D map independently. Their displayed scale is 0 to 1; the overlay uses fixed limits and 55% opacity.
- Brighter colours mean higher normalised values for that method and volume. Because the scale is relative, values cannot compare absolute importance across methods or cases.
- Signed relevance cannot be recovered from these normalised arrays. A dark voxel is not necessarily negative evidence, and a bright voxel is not necessarily clinically important tissue.
- The displayed slice may not contain the largest attribution value in the whole volume.

## What differs between the methods?

| Column | Computation | Important limit |
|---|---|---|
| Grad-CAM | Gradient-weighted deepest feature maps, followed by ReLU and interpolation from 4³ to 64³ | Coarse, smooth maps are expected from the low-resolution bottleneck |
| Zennit EpsilonPlusFlat | Layer-wise redistribution using the named composite | Architecture-specific rule/canonizer validation remains incomplete |
| Zennit EpsilonGammaBox | Gamma/box composite with input bounds [-1, 1] | Same architecture-validation limit |
| Zennit EpsilonAlpha2Beta1 | Composite with positive/negative contribution rules | Display normalisation does not preserve signed interpretation |

Zennit is the attribution library; these are three LRP-style composites, not three separately trained classifiers. The maps target the model's predicted class. The model weights stay fixed during attribution.

A convincing-looking heatmap does not establish faithful explanation. In the [perturbation results](results.md#attribution-and-perturbation-metrics), random deletion caused larger confidence drops than these methods under the current mean-masking protocol. Rule handling and masking controls need further validation.

## A misclassified example

![Misclassified test case 0 with a Zennit map, Grad-CAM and MC-dropout probabilities](figures/inference_example.png)

Test case **0** has the lower-rating label, but the checkpoint predicts the higher-rating class with probability **0.6453**. Its heatmaps therefore explain an incorrect benchmark prediction. This is the separately documented `src/infer.py --index 0 --mc-passes 20 --seed 2026` example. Its figure uses that demo's own colour scheme; do not compare its colours with the gallery scale. [Exact inference output](../results/inference_example.json).

## Data and reproducibility

- [Saved CT volumes, labels, predictions and maps](../results/qualitative.npz)
- [Verified test identities, probabilities and source hash](../results/attribution_examples.json)
- [Attribution implementation](../src/explain.py)
- [Figure renderer](../scripts/attribution_figures.py)

Run `python scripts/attribution_figures.py` from the repository root. NumPy and Matplotlib are sufficient; no model download or training is required to display the saved maps. This re-rendering changes only presentation, not the underlying attribution values.
