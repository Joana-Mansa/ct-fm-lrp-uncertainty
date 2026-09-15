[Overview](../README.md) · [Data](data.md) · [Methods](methods.md) · [Results](results.md) · [Run it](reproduce.md) · [Verification](verification.md)

# Run and reproduce

## 1. Inspect the included examples

From the repository root, Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
python scripts/preview_data.py
```

This reads `examples/data_samples.npz` and writes `docs/figures/data_samples.png`. It needs no GPU, weights or full dataset. See the [sample attribution and manifest](../examples/README.md).

## 2. Set up model inference

```bash
pip install -r requirements.txt
python scripts/download_checkpoints.py
```

The download script retrieves the audited seed-0 pretrained and scratch classifiers (about 622 MB total) from the [verified release](https://github.com/Joana-Mansa/ct-fm-lrp-uncertainty/releases/tag/verified-2026-09-15). It checks SHA-256 against [`checkpoints.json`](../checkpoints.json) and refuses to replace a different existing file.

The first model run downloads the full NoduleMNIST3D64 archive (277 MiB). By default it is stored in `data/medmnist`. To reuse an existing archive:

```bash
export MEDMNIST_ROOT=/absolute/path/to/medmnist
```

The model constructor also uses the official Hugging Face CT-FM checkpoint via `lighter_zoo`, including when loading a downstream classifier. Allow an additional download of roughly 311 MB and cache space. The default cache is `data/hf`; set `CTFM_CACHE` to use another directory.

## 3. Run one example

```bash
python src/infer.py --index 0 --mc-passes 20 --seed 2026
```

Outputs: `results/inference_example.json` and `docs/figures/inference_example.png`. These replace the previous demo outputs. The attribution maps are exploratory; see the architecture-validation limits in the methods guide.

CUDA is selected if available; otherwise the scripts use CPU. Model checks were run on an NVIDIA A100 80 GB. CPU inference is supported by the code but full CPU runtime and minimum GPU memory were not benchmarked. The 3D attribution pipeline can be memory intensive.

## 4. Repeat the verification

```bash
python scripts/verify_checkpoints.py
```

This writes to `verification_run/`, leaving the published experimental results intact. It evaluates available classifiers on all 310 inputs, checks two attribution cases, recomputes saved-array statistics, and bootstraps the seed-0 AUC difference. Compare with [`results/verification.json`](../results/verification.json).

The release includes the seed-0 pair. Verification recomputes the four-model ensemble only when all four member checkpoints are present; otherwise it explicitly skips that step. The original audit included all six available full-data checkpoints, and their per-case predictions are published in `results/checkpoint_predictions.npz`.

## 5. Train new models (optional)

Training writes experiment records and weights. Use a separate clone or back up published result files first. A new training run is not expected to reproduce unrecorded original RNG state exactly.

```bash
python src/train.py --seed 0 --tag _rerun0
python src/train.py --seed 1 --tag _rerun1
python src/train.py --seed 2 --tag _rerun2
```

Read each script's `--help` before changing settings. For CT-FM, tagged training outputs require explicit checkpoint selection or renaming in a separate analysis workspace; the analysis scripts use the seed-0 default filenames.

## Verified environment and reproducibility limits

The checkpoint audit used Python 3.11, PyTorch 2.14.0+cu130, MONAI 1.6.0, MedMNIST 3.0.2, NumPy 2.4.6 and SciPy 1.17.1. CT-FM also used `lighter-zoo` 0.1.3 and Zennit 1.0.0. [`verified-environment.json`](../results/verified-environment.json) records the environment actually used. Dependency ranges in `requirements.txt` are installation bounds, not a claim that every supported combination was tested.

Full model training was not rerun during this audit. Original stochastic analysis outputs were not all accompanied by saved RNG states. Checkpoint re-evaluation, arithmetic checks and a newly seeded repeat are different levels of evidence; see [verification](verification.md).
