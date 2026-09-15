"""Deep-ensemble uncertainty, and how it compares with MC dropout.

Every full-label run of `train.py` saves its best checkpoint under a tagged
name, so the seed replicates that produce the ablation error bars double as
ensemble members at no extra cost. This script collects whatever members exist
and compares three things against the single-model MC-dropout estimate:

  calibration     expected calibration error
  discrimination  AUC of the ensemble mean against the best single member
  uncertainty     predictive entropy, and how much of it is disagreement
                  between members rather than ambiguity in the data

The last one is the reason to bother. MC dropout perturbs one set of weights and
tends to report low mutual information, because the samples are variations on a
single solution. Independently trained members can disagree, and that
disagreement is the part of the uncertainty a practitioner can act on: it says
the evidence supports more than one model, not merely that this image is hard.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import CLASSES, loader
from model import NoduleClassifier
from train import auc_score
from uncertainty import (ensemble_predict, expected_calibration_error,
                         mc_dropout_predict)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights"


def find_members():
    """Every full-label pretrained checkpoint on disk, in a stable order."""
    return sorted(WEIGHTS.glob("ctfm_pretrained*.pt"))


def load_member(path, dev):
    model = NoduleClassifier(pretrained=True).to(dev)
    model.load_state_dict(torch.load(path, map_location=dev)["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def member_probs(model, dl, dev):
    out = []
    for x, _ in dl:
        out.append(F.softmax(model(x.to(dev)), dim=1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--mc-passes", type=int, default=20)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS.mkdir(exist_ok=True)

    paths = find_members()
    if len(paths) < 2:
        print(f"only {len(paths)} member(s) on disk, need at least 2")
        return
    print(f"found {len(paths)} ensemble members:")
    for p in paths:
        print(f"  {p.name}")

    test = loader("test", args.batch_size)
    labels = np.concatenate([y.numpy() for _, y in test])

    per_member, member_auc = [], []
    for p in paths:
        m = load_member(p, dev)
        probs = member_probs(m, test, dev)
        per_member.append(probs)
        member_auc.append(auc_score(labels, probs[:, 1]))
        print(f"  {p.name}: AUC {member_auc[-1]:.4f}")
        del m
        torch.cuda.empty_cache()

    stack = np.stack(per_member)
    mean = stack.mean(0)

    def entropy(p, eps=1e-12):
        return -(p * np.log(p + eps)).sum(1)

    ens_entropy = entropy(mean)
    expected_entropy = np.stack([entropy(s) for s in stack]).mean(0)
    mutual_info = ens_entropy - expected_entropy

    ens_auc = auc_score(labels, mean[:, 1])
    ens_ece, _ = expected_calibration_error(mean, labels)

    # Single-model MC dropout on the same test set, for comparison.
    single = load_member(paths[0], dev)
    mc_probs, mc_ent, mc_mi = [], [], []
    for x, _ in test:
        u = mc_dropout_predict(single, x.to(dev), passes=args.mc_passes)
        mc_probs.append(u["mean_probs"])
        mc_ent.append(u["entropy"])
        mc_mi.append(u["mutual_information"])
    mc_probs = np.concatenate(mc_probs)
    mc_ent = np.concatenate(mc_ent)
    mc_mi = np.concatenate(mc_mi)
    mc_auc = auc_score(labels, mc_probs[:, 1])
    mc_ece, _ = expected_calibration_error(mc_probs, labels)

    report = {
        "members": [p.name for p in paths],
        "n_members": len(paths),
        "per_member_auc": {p.name: a for p, a in zip(paths, member_auc)},
        "ensemble": {
            "auc": ens_auc,
            "ece": ens_ece,
            "mean_entropy": float(ens_entropy.mean()),
            "mean_mutual_information": float(mutual_info.mean()),
            "mi_share_of_entropy": float(mutual_info.mean() / max(ens_entropy.mean(), 1e-12)),
        },
        "mc_dropout_single_model": {
            "passes": args.mc_passes,
            "auc": mc_auc,
            "ece": mc_ece,
            "mean_entropy": float(mc_ent.mean()),
            "mean_mutual_information": float(mc_mi.mean()),
            "mi_share_of_entropy": float(mc_mi.mean() / max(mc_ent.mean(), 1e-12)),
        },
    }
    (RESULTS / "ensemble.json").write_text(json.dumps(report, indent=2))
    np.savez(RESULTS / "ensemble_per_sample.npz", labels=labels,
             ensemble_probs=mean, ensemble_entropy=ens_entropy,
             ensemble_mi=mutual_info, mc_probs=mc_probs, mc_entropy=mc_ent,
             mc_mi=mc_mi)

    print()
    print(f"{'':22} {'AUC':>7} {'ECE':>7} {'entropy':>8} {'MI':>8} {'MI share':>9}")
    print(f"{'best single member':22} {max(member_auc):>7.4f}")
    print(f"{'MC dropout (1 model)':22} {mc_auc:>7.4f} {mc_ece:>7.4f} "
          f"{mc_ent.mean():>8.4f} {mc_mi.mean():>8.4f} "
          f"{report['mc_dropout_single_model']['mi_share_of_entropy']:>9.3f}")
    print(f"{'deep ensemble':22} {ens_auc:>7.4f} {ens_ece:>7.4f} "
          f"{ens_entropy.mean():>8.4f} {mutual_info.mean():>8.4f} "
          f"{report['ensemble']['mi_share_of_entropy']:>9.3f}")


if __name__ == "__main__":
    main()
