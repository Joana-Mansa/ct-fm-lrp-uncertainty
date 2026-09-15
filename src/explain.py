"""Layer-wise relevance propagation on the CT-FM classifier, with a Grad-CAM baseline.

Why LRP rather than Grad-CAM alone. Grad-CAM is the method most medical imaging
papers reach for, and it produces a map at the resolution of the last
convolutional feature map, which for a network downsampling by 16 means a very
coarse picture. LRP redistributes the output score back through the network
layer by layer under conservation rules, so it returns a map at input
resolution. Zennit implements those rules as gradient hooks registered by layer
type, and its type registry covers Conv3d and the 3D pooling layers, so it runs
on a volumetric network without modification.

Three composites are compared because they encode different assumptions:

  EpsilonPlusFlat       positive contributions only in the convolutional stack,
                        flat redistribution in the first layer
  EpsilonGammaBox       gamma rule favouring positive weights, box rule at the
                        input, which suits bounded inputs like ours in [-1, 1]
  EpsilonAlpha2Beta1    separates positive and negative contributions at a fixed
                        ratio

Reporting one composite and calling it "the LRP explanation" hides how much the
choice of rule matters, which is the point of comparing them.
"""

import numpy as np
import torch
import torch.nn.functional as F
from zennit.attribution import Gradient
from zennit.composites import (EpsilonAlpha2Beta1, EpsilonGammaBox,
                               EpsilonPlusFlat)

COMPOSITES = {
    "epsilon_plus_flat": lambda: EpsilonPlusFlat(),
    "epsilon_gamma_box": lambda: EpsilonGammaBox(low=-1.0, high=1.0),
    "epsilon_alpha2_beta1": lambda: EpsilonAlpha2Beta1(),
}


def lrp_attribution(model, x, target, composite_name="epsilon_plus_flat"):
    """Relevance map for `target` class, same spatial shape as the input.

    Returns a numpy array of shape (N, D, H, W) with the channel summed out.
    """
    model.eval()
    composite = COMPOSITES[composite_name]()
    x = x.clone().requires_grad_(True)
    one_hot = F.one_hot(target, num_classes=2).float()

    with Gradient(model=model, composite=composite) as attributor:
        _, relevance = attributor(x, one_hot)

    return relevance.sum(1).detach().cpu().numpy()


def gradcam_attribution(model, x, target, upsample=True):
    """Grad-CAM on the encoder bottleneck, upsampled back to input resolution.

    The hook goes on the deepest encoder stage, 512 channels at 4^3. Hooking the
    SegResNet output instead would put Grad-CAM on a 2-channel map at full
    resolution, which is not what the method is defined on and produces a map
    with almost no contrast.
    """
    model.eval()
    feats = {}

    def hook(_, __, output):
        out = output[-1] if isinstance(output, (list, tuple)) else output
        feats["value"] = out
        out.retain_grad()

    handle = model.encoder.register_forward_hook(hook)
    x = x.clone().requires_grad_(True)
    logits = model(x)
    score = logits.gather(1, target.view(-1, 1)).sum()
    model.zero_grad(set_to_none=True)
    score.backward()
    handle.remove()

    act = feats["value"]
    grad = act.grad
    weights = grad.mean(dim=(2, 3, 4), keepdim=True)
    cam = F.relu((weights * act).sum(1, keepdim=True))
    if upsample:
        cam = F.interpolate(cam, size=x.shape[2:], mode="trilinear", align_corners=False)
    return cam.squeeze(1).detach().cpu().numpy()


def normalise(maps):
    """Scale each map in a batch to [0, 1] for comparison and display."""
    out = np.empty_like(maps, dtype=np.float32)
    for i, m in enumerate(maps):
        lo, hi = m.min(), m.max()
        out[i] = (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)
    return out
