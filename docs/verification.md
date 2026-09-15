[Overview](../README.md) · [Data](data.md) · [Methods](methods.md) · [Results](results.md) · [Run it](reproduce.md) · [Verification](verification.md)

# Verification record

Audit date: **15 September 2026**. The original experiment records are retained, and new verification artifacts are stored separately.

| Evidence | Check performed | Scope |
|---|---|---|
| Data | Archive MD5 against MedMNIST metadata; counts, shapes and sample indices | Original benchmark archive |
| Architecture | Count parameters; strict loading; bottleneck shape | 77,763,042 parameters; 512 × 4 × 4 × 4 per case |
| Classification | Full inference from six saved checkpoints | 310 test patches each |
| Matched comparisons | Pair runs by training budget and seed | Three completed recorded pairs at every budget; seed-1 scratch checkpoint unavailable |
| Ensemble | Average four checkpoint probability arrays | AUC and ECE match saved summary |
| Explanations | Finite-map/shape and perturbation endpoint smoke checks | Two cases, not full LRP validation |
| Saved-array statistics | Recompute ECE, AOPC and tie-aware Spearman with p-values | Original 64-case analysis |
| Sampling uncertainty | Paired stratified bootstrap of seed-0 AUC difference | 2,000 patch-level replicates |

## Material corrections

- Corrected 87.2M to 77.8M classifier parameters.
- Clarified that nodule labels are rating-derived and pretraining belongs to the original CT-FM authors.
- Fixed figure aggregation to compare matched seeds only. Completed GitHub records give mean gains +0.0164, +0.0136 and +0.0058.
- Distinguished three recorded full-data pairs from the two pairs with both checkpoints available.
- Corrected deletion baseline from dataset mean to current batch mean, and stability sample size from 64 to 16.
- Distinguished finite Zennit maps from architecture-validated LRP.
- Removed claims that low head-dropout mutual information proves aleatoric uncertainty or reliability.
- Added the confidence interval, unadjusted correlation p-values and unfavourable individual runs.

## Traceability

[`results/verification.json`](../results/verification.json) records checkpoint hashes, source hashes at evaluation time and recomputed metrics. [`checkpoints.json`](../checkpoints.json) identifies the distributed weights. [`examples/data_manifest.json`](../examples/data_manifest.json) records the archive hashes and example provenance. Source hashes describe the code used during the audit, before subsequent documentation, path-portability and reporting fixes; they are not asserted to hash the final edited source tree.

The portable [`scripts/verify_checkpoints.py`](../scripts/verify_checkpoints.py) makes a new verification record in `verification_run/`. Full training, patient-level leakage auditing, clinical evaluation and external validation remain outside the completed checks.

## Reconciliation with newer GitHub records

The server checkout was behind GitHub. Completed records at GitHub commit `0f9d732` add the full seed-1 pair and change some lower-budget values compared with the initial server files. The top-level ablation records and paired plot now use the completed GitHub versions. [Original server records](../results/server-snapshot/README.md) are retained separately because their relationship to the completed records cannot be established from metadata alone.

The six-checkpoint audit remains unchanged. All seed-0 and seed-2 full-data metrics match their completed records. Seed-1 pretrained AUC differs by 0.000064 between its completed record and checkpoint re-evaluation; accuracy and balanced accuracy agree. Its scratch checkpoint is unavailable. Lower-budget metrics are supported by experiment records, not independent checkpoint re-evaluation.
