"""Run the explanation and uncertainty analysis on the trained model.

Three results come out of this script.

1. Faithfulness of each attribution method against a random-order baseline.
   Without the baseline the deletion numbers cannot be read, because their scale
   depends on the model and the data rather than on the explanation.

2. Stability of each map under small input perturbations.

3. The relationship between uncertainty and faithfulness. For every test volume
   we record predictive entropy from MC dropout and the per-sample AOPC of its
   attribution, then take the rank correlation between them. A negative
   correlation means explanations get less faithful as the model gets less sure.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import CLASSES, Nodules
from explain import COMPOSITES, gradcam_attribution, lrp_attribution, normalise
from faithfulness import deletion_curve, insertion_curve, stability, summarise
from model import NoduleClassifier
from uncertainty import (expected_calibration_error, mc_dropout_predict,
                         spearman)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights"


def load_model(dev, pretrained=True):
    name = "ctfm_pretrained.pt" if pretrained else "ctfm_scratch.pt"
    model = NoduleClassifier(pretrained=pretrained).to(dev)
    model.load_state_dict(torch.load(WEIGHTS / name, map_location=dev)["state_dict"])
    model.eval()
    return model


def attribute(model, x, target, method):
    if method == "gradcam":
        return gradcam_attribution(model, x, target)
    return lrp_attribution(model, x, target, method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="test volumes to analyse")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--mc-passes", type=int, default=20)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS.mkdir(exist_ok=True)
    model = load_model(dev, pretrained=True)

    ds = Nodules("test")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    x_all = torch.stack([ds[i][0] for i in idx])
    y_all = torch.tensor([ds[i][1] for i in idx])

    methods = list(COMPOSITES) + ["gradcam"]
    per_method = {m: {"del_auc": [], "ins_auc": [], "aopc": []} for m in methods}
    rand_del_auc, rand_aopc = [], []
    entropies, mi, mean_probs = [], [], []

    for start in range(0, len(x_all), args.batch):
        x = x_all[start:start + args.batch].to(dev)
        y = y_all[start:start + args.batch].to(dev)
        with torch.no_grad():
            pred = model(x).argmax(1)

        unc = mc_dropout_predict(model, x, passes=args.mc_passes)
        entropies.append(unc["entropy"])
        mi.append(unc["mutual_information"])
        mean_probs.append(unc["mean_probs"])

        for m in methods:
            a = attribute(model, x, pred, m)
            d = deletion_curve(model, x, pred, a, steps=args.steps)
            i = insertion_curve(model, x, pred, a, steps=args.steps)
            per_method[m]["del_auc"].append(
                np.trapezoid(d, dx=1 / args.steps, axis=1))
            per_method[m]["ins_auc"].append(
                np.trapezoid(i, dx=1 / args.steps, axis=1))
            per_method[m]["aopc"].append((d[:, :1] - d).mean(axis=1))

        r = deletion_curve(model, x, pred, np.zeros(x.shape[0:1] + x.shape[2:]),
                           steps=args.steps, order="random")
        rand_del_auc.append(np.trapezoid(r, dx=1 / args.steps, axis=1))
        rand_aopc.append((r[:, :1] - r).mean(axis=1))
        print(f"  analysed {min(start + args.batch, len(x_all))}/{len(x_all)}",
              flush=True)

    entropies = np.concatenate(entropies)
    mi = np.concatenate(mi)
    mean_probs = np.concatenate(mean_probs)
    rand_del_auc = np.concatenate(rand_del_auc)
    rand_aopc = np.concatenate(rand_aopc)

    faith = {"random_baseline": {"deletion_auc": float(rand_del_auc.mean()),
                                 "aopc": float(rand_aopc.mean())}}
    for m in methods:
        d = np.concatenate(per_method[m]["del_auc"])
        i = np.concatenate(per_method[m]["ins_auc"])
        a = np.concatenate(per_method[m]["aopc"])
        faith[m] = {
            "deletion_auc": float(d.mean()),
            "insertion_auc": float(i.mean()),
            "aopc": float(a.mean()),
            "aopc_over_random": float(a.mean() - rand_aopc.mean()),
            "uncertainty_vs_aopc_spearman": spearman(entropies, a),
        }
        per_method[m]["aopc_flat"] = a.tolist()

    stab = {}
    x_small = x_all[:min(16, len(x_all))].to(dev)
    with torch.no_grad():
        t_small = model(x_small).argmax(1)
    for m in methods:
        stab[m] = stability(model, x_small, t_small,
                            lambda xx, tt, m=m: attribute(model, xx, tt, m))

    ece, ece_detail = expected_calibration_error(mean_probs, y_all.numpy())

    (RESULTS / "explanations.json").write_text(json.dumps({
        "n_analysed": len(x_all),
        "deletion_steps": args.steps,
        "mc_passes": args.mc_passes,
        "faithfulness": faith,
        "stability_corr": stab,
        "calibration": {"ece": ece, "bins": ece_detail},
        "uncertainty": {
            "entropy_mean": float(entropies.mean()),
            "entropy_median": float(np.median(entropies)),
            "mutual_information_mean": float(mi.mean()),
        },
    }, indent=2))

    np.savez(RESULTS / "per_sample.npz", entropy=entropies, mi=mi,
             labels=y_all.numpy(), probs=mean_probs,
             **{f"aopc_{m}": np.asarray(per_method[m]["aopc_flat"]) for m in methods})

    # A small qualitative panel for the README.
    x_vis = x_all[:4].to(dev)
    with torch.no_grad():
        t_vis = model(x_vis).argmax(1)
    maps = {m: normalise(attribute(model, x_vis, t_vis, m)) for m in methods}
    np.savez(RESULTS / "qualitative.npz", volume=x_vis.cpu().numpy(),
             label=y_all[:4].numpy(), pred=t_vis.cpu().numpy(), **maps)

    print(json.dumps(faith, indent=2))


if __name__ == "__main__":
    main()
