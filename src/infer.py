"""Run the trained model on a single CT volume and report everything it knows.

This is the end-to-end path a user would actually take: give it one volume, get
back the prediction, how sure the model is, and what it looked at. Everything
here is read off the trained checkpoint, so it needs no training run.

The output figure carries three things side by side, because none of them is
enough on its own. A prediction without an uncertainty is a number with no error
bar. An uncertainty without an explanation says the model is unsure but not
where. An explanation without either invites the reader to trust a heatmap for a
call the model was never confident about.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data import CLASSES, Nodules
from explain import gradcam_attribution, lrp_attribution, normalise
from model import NoduleClassifier
from uncertainty import mc_dropout_predict

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights"
FIGS = ROOT / "docs" / "figures"


def load_model(dev, pretrained=True):
    name = "ctfm_pretrained.pt" if pretrained else "ctfm_scratch.pt"
    model = NoduleClassifier(pretrained=pretrained).to(dev)
    model.load_state_dict(torch.load(WEIGHTS / name, map_location=dev)["state_dict"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0, help="test volume to explain")
    ap.add_argument("--method", default="epsilon_plus_flat")
    ap.add_argument("--mc-passes", type=int, default=30)
    ap.add_argument("--out", default="inference_example.png")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    FIGS.mkdir(parents=True, exist_ok=True)

    ds = Nodules("test")
    x = ds[args.index][0].unsqueeze(0).to(dev)
    truth = ds[args.index][1]

    model = load_model(dev)

    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())

    unc = mc_dropout_predict(model, x, passes=args.mc_passes)
    entropy = float(unc["entropy"][0])
    mutual_info = float(unc["mutual_information"][0])
    spread = float(unc["std"][0])

    target = torch.tensor([pred], device=dev)
    lrp = normalise(lrp_attribution(model, x, target, args.method))[0]
    cam = normalise(gradcam_attribution(model, x, target))[0]

    report = {
        "index": args.index,
        "ground_truth": CLASSES[truth],
        "prediction": CLASSES[pred],
        "correct": bool(pred == truth),
        "probabilities": {CLASSES[i]: float(p) for i, p in enumerate(probs)},
        "mc_dropout": {
            "passes": args.mc_passes,
            "mean_probabilities": {CLASSES[i]: float(p)
                                   for i, p in enumerate(unc["mean_probs"][0])},
            "predictive_entropy": entropy,
            "mutual_information": mutual_info,
            "max_std_over_passes": spread,
        },
        "attribution_method": args.method,
    }
    (RESULTS / "inference_example.json").write_text(json.dumps(report, indent=2))

    print(f"volume {args.index}: truth {CLASSES[truth]}, predicted {CLASSES[pred]} "
          f"at p={probs[pred]:.3f}")
    print(f"  predictive entropy {entropy:.4f}, mutual information {mutual_info:.4f}")
    print(f"  spread over {args.mc_passes} dropout passes: {spread:.4f}")

    vol = x[0, 0].cpu().numpy()
    mid = vol.shape[0] // 2
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))

    axes[0].imshow(vol[mid], cmap="gray")
    axes[0].set_title(f"CT, central slice\ntruth: {CLASSES[truth]}", fontsize=9)

    axes[1].imshow(vol[mid], cmap="gray")
    axes[1].imshow(lrp[mid], cmap="jet", alpha=0.45)
    axes[1].set_title(f"LRP ({args.method})", fontsize=9)

    axes[2].imshow(vol[mid], cmap="gray")
    axes[2].imshow(cam[mid], cmap="jet", alpha=0.45)
    axes[2].set_title("Grad-CAM", fontsize=9)

    axes[3].bar(CLASSES, unc["mean_probs"][0], color=["#3b6ea5", "#b5651d"],
                yerr=[spread, spread], capsize=4)
    axes[3].set_ylim(0, 1)
    axes[3].set_title(f"MC dropout, {args.mc_passes} passes\n"
                      f"entropy {entropy:.3f}", fontsize=9)
    axes[3].set_ylabel("probability")

    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(FIGS / args.out, dpi=140)
    print(f"  wrote {FIGS / args.out}")


if __name__ == "__main__":
    main()
