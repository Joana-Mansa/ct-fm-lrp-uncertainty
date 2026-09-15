"""Fine-tune CT-FM on lung nodule malignancy, and run the label-efficiency ablation.

The ablation is the point of this script. Self-supervised pretraining is only
worth claiming if it beats the same architecture trained from scratch, and the
place it should show most is where labels are scarce. So every configuration is
trained twice, once from CT-FM weights and once from random initialisation, at
several fractions of the training set.

Because the dataset is imbalanced we track balanced accuracy and AUC. Model
selection uses validation AUC.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import loader
from model import NoduleClassifier

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
WEIGHTS = ROOT / "weights"


def auc_score(y, p):
    """ROC AUC via the rank statistic, so scikit-learn is not needed."""
    y = np.asarray(y)
    p = np.asarray(p)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    # Average ranks over ties so the statistic is correct for repeated scores.
    _, inv, counts = np.unique(p, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * neg.sum()))


@torch.no_grad()
def evaluate(model, dl, dev):
    """Accuracy, balanced accuracy and AUC on one split."""
    model.eval()
    probs, labels = [], []
    for x, y in dl:
        p = F.softmax(model(x.to(dev)), dim=1)[:, 1]
        probs.append(p.cpu().numpy())
        labels.append(y.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    pred = (probs >= 0.5).astype(int)
    acc = float((pred == labels).mean())
    recalls = [float((pred[labels == c] == c).mean()) for c in (0, 1)
               if (labels == c).sum() > 0]
    return {"accuracy": acc, "balanced_accuracy": float(np.mean(recalls)),
            "auc": auc_score(labels, probs)}


def run_one(pretrained, fraction, args, dev, tag):
    """Train a single arm of the ablation and return its test metrics."""
    train = loader("train", args.batch_size, fraction=fraction, seed=args.seed)
    val = loader("val", args.batch_size)
    test = loader("test", args.batch_size)

    torch.manual_seed(args.seed)
    model = NoduleClassifier(pretrained=pretrained).to(dev)
    opt = torch.optim.AdamW(model.param_groups(args.lr, args.head_lr),
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    # Class weights counter the benign majority.
    counts = np.bincount(train.dataset.labels, minlength=2)
    w = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)),
                     dtype=torch.float32, device=dev)

    best_auc, best_state, history = -1.0, None, []
    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        for x, y in train:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y, weight=w)
            loss.backward()
            opt.step()
            running += loss.item() * x.shape[0]
            seen += x.shape[0]
        sched.step()
        v = evaluate(model, val, dev)
        history.append({"epoch": epoch + 1, "loss": running / seen, **v})
        if v["auc"] > best_auc:
            best_auc = v["auc"]
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        print(f"  [{tag}] epoch {epoch + 1:2d}/{args.epochs} loss {running / seen:.4f}"
              f" val auc {v['auc']:.4f}", flush=True)

    model.load_state_dict(best_state)
    t = evaluate(model, test, dev)
    print(f"  [{tag}] test acc {t['accuracy']:.4f} bal {t['balanced_accuracy']:.4f}"
          f" auc {t['auc']:.4f}", flush=True)

    if fraction == 1.0:
        WEIGHTS.mkdir(exist_ok=True)
        name = "ctfm_pretrained.pt" if pretrained else "ctfm_scratch.pt"
        torch.save({"state_dict": best_state, "pretrained": pretrained}, WEIGHTS / name)

    return {"n_train": len(train.dataset), "val_auc": best_auc,
            "test": t, "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5, help="encoder lr")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 1.0])
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS.mkdir(exist_ok=True)

    out, start = [], time.time()
    for fraction in args.fractions:
        for pretrained in (True, False):
            tag = f"{'ctfm' if pretrained else 'scratch'} {int(fraction * 100)}%"
            print(f"=== {tag} ===", flush=True)
            r = run_one(pretrained, fraction, args, dev, tag)
            r.update({"fraction": fraction, "pretrained": pretrained})
            out.append(r)
            (RESULTS / "ablation.json").write_text(json.dumps(
                {"epochs": args.epochs, "encoder_lr": args.lr,
                 "head_lr": args.head_lr, "seed": args.seed,
                 "minutes": (time.time() - start) / 60, "runs": out}, indent=2))

    print(f"done in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
