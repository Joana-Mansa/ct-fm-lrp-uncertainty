[Overview](../README.md) · [Data](data.md) · [Methods](methods.md) · [Results](results.md) · [Run it](reproduce.md) · [Verification](verification.md)

# Data and examples

NoduleMNIST3D contains CT nodule patches derived from LIDC-IDRI. The binary target groups malignancy ratings 1/2 and 4/5; rating 3 is excluded in benchmark construction. The labels encode radiologist assessments, not pathology-confirmed cancer.

## Verified archive

Official [MedMNIST dataset metadata](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py) identifies the [64-pixel archive](https://zenodo.org/records/10519652/files/nodulemnist3d_64.npz?download=1). Its MD5 matched the published value during the audit.

| Split | Image array shape | Storage |
|---|---|---|
| train | 1158 × 64 × 64 × 64 | uint8 |
| val | 165 × 64 × 64 × 64 | uint8 |
| test | 310 × 64 × 64 × 64 | uint8 |

Images are converted using `image.astype(float32) / 127.5 - 1.0`, then a channel dimension is added. These arrays are resized, quantised benchmark images, not raw Hounsfield-unit volumes or DICOM series.

| Split | Label 0 (lower ratings) | Label 1 (higher ratings) |
|---|---:|---:|
| Train | 863 | 295 |
| Validation | 123 | 42 |
| Test | 246 | 64 |

The code retains the benchmark short labels `benign` and `malignant`. Read these as rating-derived categories throughout the JSON and figures. The sample panel shows array axes 0, 1 and 2 at index 32; anatomical orientation is not established by these arrays.

## Inspect a real example

[Sample archive and attribution](../examples/README.md) · [Exact indices, counts and hashes](../examples/data_manifest.json)

The sample archive contains unmodified uint8 images, labels and their original test indices. Examples are chosen by first occurrence of each class, not by model performance. Run `python scripts/preview_data.py` from the repository root to recreate the figure.

## What cannot be audited from this archive

The distributed NPZ does not include patient or site identifiers. This project uses the supplied train/validation/test partition, but does not independently establish patient-level separation, scanner generalisation or external validation. CT-FM was pretrained on Imaging Data Commons scans; possible overlap with the source collection was not audited. The project input scaling also differs from the original [CT-FM model-card preprocessing](https://huggingface.co/project-lighter/ct_fm_feature_extractor).

## Attribution and reuse

MedMNIST v2 data authors retain credit; the included examples are redistributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The panel changes presentation only. Dataset licensing is distinct from model and source-code licensing.
