"""The classifier built on the CT-FM encoder.

CT-FM is a SegResNet pretrained with contrastive self-supervised learning on
148,000 CT scans from the Imaging Data Commons. We take its encoder and put a
small classification head on top.

Two details worth recording, because both cost time to find.

The published checkpoint does not load into MONAI's SegResNetDS. The parameter
names differ and `load_state_dict(strict=False)` matches nothing, which fails
silently and trains a randomly initialised network while the code looks correct.
The `lighter_zoo` loader is the one that works, so that is what we use, and the
from-scratch arm of the ablation is built through the same class with pretrained
weights left out. That keeps the two arms architecturally identical.

CT-FM downsamples by 16, so input side lengths must be divisible by 16. A 28^3
volume raises an error. This is why the 64^3 release of the dataset is used.
"""

import torch
import torch.nn as nn
from lighter_zoo import SegResNet

HF_ID = "project-lighter/ct_fm_feature_extractor"
CACHE = "data/hf"


def build_encoder(pretrained=True, cache_dir=CACHE):
    """CT-FM encoder, with or without the pretrained weights.

    The from-scratch arm is the same object with every parameter reinitialised,
    rather than a separately constructed network. Building a second network from
    a guessed config gave an architecture with 19.9M parameters against the real
    87.2M, which would have made the ablation compare two different models and
    report the difference as an effect of pretraining.
    """
    model = SegResNet.from_pretrained(HF_ID, cache_dir=cache_dir)
    if pretrained:
        return model
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm)):
            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
    return model


class NoduleClassifier(nn.Module):
    """CT-FM encoder with a dropout head, for benign against malignant.

    Dropout stays active at inference time for the MC-dropout uncertainty
    estimate, so it is a named module rather than a functional call.
    """

    def __init__(self, pretrained=True, n_classes=2, p_drop=0.3, cache_dir=CACHE):
        super().__init__()
        self.encoder = build_encoder(pretrained=pretrained, cache_dir=cache_dir)
        with torch.no_grad():
            probe = self.encoder(torch.zeros(1, 1, 64, 64, 64))
            probe = probe[0] if isinstance(probe, (list, tuple)) else probe
            n_feat = probe.shape[1]
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(n_feat, n_classes)

    def features(self, x):
        f = self.encoder(x)
        return f[0] if isinstance(f, (list, tuple)) else f

    def forward(self, x):
        h = self.pool(self.features(x)).flatten(1)
        return self.fc(self.drop(h))


def enable_mc_dropout(model):
    """Put the model in eval mode but leave dropout sampling."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    return model
