"""Build every figure used in the README from the saved result files."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 140, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

CTFM = "#3b6ea5"
SCRATCH = "#b5651d"
NAMES = {"epsilon_plus_flat": "LRP eps+flat", "epsilon_gamma_box": "LRP eps-gamma-box",
         "epsilon_alpha2_beta1": "LRP a2b1", "gradcam": "Grad-CAM"}


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def collect_ablation():
    """Use only matched pretrained/scratch runs from the same seed and budget."""
    pairs = {}
    for p in sorted(RESULTS.glob("ablation*.json")):
        document = json.loads(p.read_text())
        for run in document.get("runs", []):
            pairs.setdefault((document["seed"], run["fraction"]), {})[run["pretrained"]] = run["test"]["auc"]
    out = {}
    for (seed, fraction), arms in sorted(pairs.items()):
        if True in arms and False in arms:
            row = out.setdefault(fraction, {"CT-FM": [], "scratch": [], "seeds": []})
            row["CT-FM"].append(arms[True])
            row["scratch"].append(arms[False])
            row["seeds"].append(seed)
    return out


def fig_ablation():
    data = collect_ablation()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    fracs = sorted(data)
    x = np.arange(len(fracs))
    for arm, colour in (("CT-FM", CTFM), ("scratch", SCRATCH)):
        means = [np.mean(data[f][arm]) for f in fracs]
        axes[0].plot(x, means, "o-", color=colour, label=arm)
        for i, f in enumerate(fracs):
            axes[0].scatter(np.full(len(data[f][arm]), i), data[f][arm], color=colour, alpha=.5, s=15)
    labels = [f"{int(f*100)}%\n{len(data[f]['seeds'])} paired seeds" for f in fracs]
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Test AUC")
    axes[0].set_title("Matched-seed label efficiency")
    axes[0].legend()
    gains = [np.mean(np.array(data[f]["CT-FM"]) - np.array(data[f]["scratch"])) for f in fracs]
    axes[1].bar(x, gains, color=CTFM)
    axes[1].set_xticks(x, labels)
    axes[1].axhline(0, color="gray", linewidth=.8)
    axes[1].set_ylabel("Mean paired AUC difference")
    axes[1].set_title("CT-FM minus scratch")
    fig.savefig(FIGS / "ablation.png")
    plt.close(fig)


def fig_faithfulness():
    d = load("explanations.json")
    if not d:
        return
    f = d["faithfulness"]
    methods = [m for m in NAMES if m in f]
    rand = f["random_baseline"]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    x = np.arange(len(methods))
    axes[0].bar(x, [f[m]["deletion_auc"] for m in methods], color=CTFM)
    axes[0].axhline(rand["deletion_auc"], color=SCRATCH, linestyle="--",
                    label="random order")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([NAMES[m] for m in methods], rotation=20, ha="right", fontsize=7)
    axes[0].set_ylabel("deletion AUC")
    axes[0].set_title("Deletion under mean masking")
    axes[0].legend(fontsize=7)

    axes[1].bar(x, [f[m]["aopc_over_random"] for m in methods], color=CTFM)
    axes[1].axhline(0, color="#444", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([NAMES[m] for m in methods], rotation=20, ha="right", fontsize=7)
    axes[1].set_ylabel("AOPC over random")
    axes[1].set_title("AOPC difference from random")
    plt.tight_layout()
    plt.savefig(FIGS / "faithfulness.png")
    plt.close()


def fig_uncertainty_vs_faithfulness():
    p = RESULTS / "per_sample.npz"
    d = load("explanations.json")
    if not p.exists() or not d:
        return
    z = np.load(p)
    methods = [m for m in NAMES if f"aopc_{m}" in z]
    fig, axes = plt.subplots(1, len(methods), figsize=(3.1 * len(methods), 3),
                             squeeze=False)
    for ax, m in zip(axes[0], methods):
        ent, aopc = z["entropy"], z[f"aopc_{m}"]
        ax.scatter(ent, aopc, s=12, alpha=0.6, color=CTFM)
        rho = d["faithfulness"][m]["uncertainty_vs_aopc_spearman"]
        ax.set_title(f"{NAMES[m]}\nspearman {rho:+.2f}", fontsize=8)
        ax.set_xlabel("predictive entropy")
        ax.set_ylabel("AOPC")
    fig.suptitle("Does the explanation get worse where the model is unsure?",
                 fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGS / "uncertainty_vs_faithfulness.png")
    plt.close()


def fig_qualitative():
    p = RESULTS / "qualitative.npz"
    if not p.exists():
        return
    z = np.load(p)
    vol = z["volume"][:, 0]
    methods = [m for m in NAMES if m in z]
    n = min(3, len(vol))
    rows = 1 + len(methods)
    fig, axes = plt.subplots(rows, n, figsize=(2.1 * n, 2.0 * rows), squeeze=False)
    mid = vol.shape[1] // 2
    for j in range(n):
        axes[0][j].imshow(vol[j, mid], cmap="gray")
        axes[0][j].set_title(f"label {z['label'][j]}  pred {z['pred'][j]}", fontsize=7)
        for i, m in enumerate(methods):
            axes[i + 1][j].imshow(vol[j, mid], cmap="gray")
            axes[i + 1][j].imshow(z[m][j, mid], cmap="jet", alpha=0.45)
    for i, lab in enumerate(["CT slice"] + [NAMES[m] for m in methods]):
        axes[i][0].set_ylabel(lab, fontsize=7)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Attribution on the central axial slice", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGS / "attribution_panel.png")
    plt.close()


def fig_calibration():
    d = load("explanations.json")
    if not d or "calibration" not in d:
        return
    bins = d["calibration"]["bins"]
    if not bins:
        return
    conf = [b["confidence"] for b in bins]
    acc = [b["accuracy"] for b in bins]
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=0.9, label="perfect")
    ax.plot(conf, acc, "o-", color=CTFM, label="MC dropout")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Calibration, ECE {d['calibration']['ece']:.3f}")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGS / "calibration.png")
    plt.close()


def fig_training_curves():
    runs = load("ablation.json")
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for r in runs["runs"]:
        if r["fraction"] != 1.0:
            continue
        h = r["history"]
        ax.plot([e["epoch"] for e in h], [e["auc"] for e in h],
                color=CTFM if r["pretrained"] else SCRATCH,
                label="CT-FM" if r["pretrained"] else "scratch")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation AUC")
    ax.set_title("Training at 100% labels")
    ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGS / "training_curves.png")
    plt.close()


def fig_ensemble():
    d = load("ensemble.json")
    p = RESULTS / "ensemble_per_sample.npz"
    if not d or not p.exists():
        return
    z = np.load(p)
    ens, mc = d["ensemble"], d["mc_dropout_single_model"]

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.1))

    labels = ["MC dropout\n(1 model)", f"deep ensemble\n({d['n_members']} members)"]
    axes[0].bar(labels, [mc["ece"], ens["ece"]], color=[SCRATCH, CTFM])
    axes[0].set_ylabel("expected calibration error")
    axes[0].set_title("Calibration, lower is better")

    axes[1].bar(labels, [mc["mi_share_of_entropy"], ens["mi_share_of_entropy"]],
                color=[SCRATCH, CTFM])
    axes[1].set_ylabel("mutual information / entropy")
    axes[1].set_title("Share of uncertainty from\nmodel disagreement")

    axes[2].hist(z["mc_entropy"], bins=30, alpha=0.6, color=SCRATCH,
                 label="MC dropout")
    axes[2].hist(z["ensemble_entropy"], bins=30, alpha=0.6, color=CTFM,
                 label="deep ensemble")
    axes[2].set_xlabel("predictive entropy")
    axes[2].set_ylabel("test volumes")
    axes[2].set_title("Entropy distribution")
    axes[2].legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGS / "ensemble.png")
    plt.close()


if __name__ == "__main__":
    for fn in [fig_ablation, fig_faithfulness, fig_uncertainty_vs_faithfulness,
               fig_qualitative, fig_calibration, fig_training_curves,
               fig_ensemble]:
        fn()
        print(f"built {fn.__name__}")
