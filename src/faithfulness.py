"""Does the attribution actually describe what the model used?

An attribution map is a claim: these voxels carried the evidence. The claim is
testable. Remove the voxels the map ranks highest and the predicted probability
for that class should fall. Remove the same number of voxels chosen at random
and it should fall much less. If it does not, the map is decoration.

Two curves are produced.

  deletion    progressively replace the highest-ranked voxels with the dataset
              mean value and track the target-class probability. A faithful map
              gives a steep early drop, so a low area under the curve is good.
  insertion   start from a blurred volume and progressively restore the
              highest-ranked voxels. A faithful map recovers the probability
              quickly, so a high area under the curve is good.

AOPC is the mean drop from the original probability across all deletion steps,
higher being more faithful. A random-order baseline is run alongside every map,
because the absolute numbers depend on the model and the data and only the gap
against random is interpretable.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _blur(x, kernel=5):
    """Heavily smoothed version of the volume, used as the insertion baseline."""
    pad = kernel // 2
    weight = torch.ones(1, 1, kernel, kernel, kernel, device=x.device) / kernel ** 3
    return F.conv3d(F.pad(x, [pad] * 6, mode="replicate"), weight)


@torch.no_grad()
def deletion_curve(model, x, target, attribution, steps=20, baseline_value=None,
                   order="descending"):
    """Target-class probability as top-ranked voxels are removed."""
    model.eval()
    n, _, d, h, w = x.shape
    n_vox = d * h * w
    flat_attr = torch.as_tensor(attribution, device=x.device).reshape(n, n_vox)

    if order == "random":
        rank = torch.rand_like(flat_attr).argsort(dim=1, descending=True)
    else:
        rank = flat_attr.argsort(dim=1, descending=True)

    base = torch.full_like(x, x.mean().item() if baseline_value is None else baseline_value)
    probs = []
    for s in range(steps + 1):
        k = int(round(n_vox * s / steps))
        cur = x.clone().reshape(n, -1)
        if k > 0:
            idx = rank[:, :k]
            cur.scatter_(1, idx, base.reshape(n, -1).gather(1, idx))
        cur = cur.reshape(n, 1, d, h, w)
        p = F.softmax(model(cur), dim=1).gather(1, target.view(-1, 1)).squeeze(1)
        probs.append(p.cpu().numpy())
    return np.stack(probs, axis=1)


@torch.no_grad()
def insertion_curve(model, x, target, attribution, steps=20):
    """Target-class probability as top-ranked voxels are restored onto a blur."""
    model.eval()
    n, _, d, h, w = x.shape
    n_vox = d * h * w
    flat_attr = torch.as_tensor(attribution, device=x.device).reshape(n, n_vox)
    rank = flat_attr.argsort(dim=1, descending=True)
    base = _blur(x)

    probs = []
    for s in range(steps + 1):
        k = int(round(n_vox * s / steps))
        cur = base.clone().reshape(n, -1)
        if k > 0:
            idx = rank[:, :k]
            cur.scatter_(1, idx, x.reshape(n, -1).gather(1, idx))
        cur = cur.reshape(n, 1, d, h, w)
        p = F.softmax(model(cur), dim=1).gather(1, target.view(-1, 1)).squeeze(1)
        probs.append(p.cpu().numpy())
    return np.stack(probs, axis=1)


def summarise(curve):
    """Area under a curve and the mean drop from its starting value."""
    auc = float(np.trapezoid(curve, dx=1.0 / (curve.shape[1] - 1), axis=1).mean())
    aopc = float((curve[:, :1] - curve).mean())
    return {"auc": auc, "aopc": aopc}


@torch.no_grad()
def stability(model, x, target, attribute_fn, sigma=0.05, repeats=3):
    """How much the attribution moves when the input is slightly perturbed.

    Reported as the mean correlation between the map on the clean input and the
    map on a noisy copy. An explanation that changes completely under noise the
    model itself ignores is not describing the model.
    """
    base = attribute_fn(x, target).reshape(len(x), -1)
    base = base - base.mean(1, keepdims=True)
    corrs = []
    for _ in range(repeats):
        noisy = x + sigma * torch.randn_like(x)
        other = attribute_fn(noisy, target).reshape(len(x), -1)
        other = other - other.mean(1, keepdims=True)
        num = (base * other).sum(1)
        den = np.sqrt((base ** 2).sum(1) * (other ** 2).sum(1)) + 1e-12
        corrs.append(num / den)
    return float(np.mean(np.stack(corrs)))
