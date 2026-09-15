"""Uncertainty estimates, and the question this repository is built around.

Two estimators, both cheap and both standard.

  MC dropout      keep dropout sampling at test time and take the spread over
                  repeated forward passes
  deep ensemble   train the same architecture several times from different
                  seeds and take the spread over members

Predictive entropy is the summary statistic in both cases.

The question worth asking is the one Salahuddin's renal cyst paper asks of a
different model: when the network is unsure, is its explanation also worse? If
faithfulness drops on the cases where uncertainty is high, then an explanation
shown to a clinician is least trustworthy exactly where they most need it, and
uncertainty should gate whether the explanation is displayed at all.
"""

import numpy as np
import torch
import torch.nn.functional as F

from model import enable_mc_dropout


def entropy(probs, eps=1e-12):
    """Predictive entropy of a probability vector, per sample."""
    return float(-(probs * np.log(probs + eps)).sum())


@torch.no_grad()
def mc_dropout_predict(model, x, passes=20):
    """Mean probabilities and predictive entropy over dropout samples."""
    enable_mc_dropout(model)
    samples = []
    for _ in range(passes):
        samples.append(F.softmax(model(x), dim=1).cpu().numpy())
    samples = np.stack(samples)
    mean = samples.mean(0)
    ent = np.array([entropy(p) for p in mean])
    # Mutual information separates what the model does not know from what the
    # data does not determine. Low total entropy with high MI means disagreement.
    expected_ent = np.array([np.mean([entropy(s[i]) for s in samples])
                             for i in range(len(mean))])
    model.eval()
    return {"mean_probs": mean, "entropy": ent,
            "mutual_information": ent - expected_ent,
            "std": samples.std(0).max(1)}


@torch.no_grad()
def ensemble_predict(models, x):
    """Mean probabilities and predictive entropy over ensemble members."""
    samples = []
    for m in models:
        m.eval()
        samples.append(F.softmax(m(x), dim=1).cpu().numpy())
    samples = np.stack(samples)
    mean = samples.mean(0)
    ent = np.array([entropy(p) for p in mean])
    expected_ent = np.array([np.mean([entropy(s[i]) for s in samples])
                             for i in range(len(mean))])
    return {"mean_probs": mean, "entropy": ent,
            "mutual_information": ent - expected_ent,
            "std": samples.std(0).max(1)}


def expected_calibration_error(probs, labels, bins=10):
    """Standard ECE over equal-width confidence bins."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    detail = []
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        acc, avg_conf = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(acc - avg_conf)
        detail.append({"bin": [float(edges[i]), float(edges[i + 1])],
                       "n": int(m.sum()), "accuracy": float(acc),
                       "confidence": float(avg_conf)})
    return float(ece), detail


def spearman(a, b):
    """Rank correlation, used to relate uncertainty to faithfulness."""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")
