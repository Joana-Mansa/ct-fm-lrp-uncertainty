"""Official CT-FM encoder with a downstream nodule classification head.

The deepest feature map is pooled; the same encoder architecture is used in
the pretrained and randomly reinitialised comparison arms.
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from lighter_zoo import SegResNet

HF_ID = "project-lighter/ct_fm_feature_extractor"
CACHE = os.environ.get("CTFM_CACHE", str(Path(__file__).resolve().parents[1] / "data" / "hf"))


def build_encoder(pretrained=True, cache_dir=CACHE):
    """Load the official encoder, optionally reinitialising the same architecture."""
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
    """CT-FM bottleneck features with a dropout head, benign against malignant.

    Dropout is a named module and stays active under `enable_mc_dropout`, which
    is how the MC-dropout uncertainty estimate is taken.
    """

    def __init__(self, pretrained=True, n_classes=2, p_drop=0.3, cache_dir=CACHE):
        super().__init__()
        full = build_encoder(pretrained=pretrained, cache_dir=cache_dir)
        self.encoder = full.encoder  # decoder discarded, it is not used here
        with torch.no_grad():
            pyramid = self.encoder(torch.zeros(1, 1, 64, 64, 64))
            n_feat = pyramid[-1].shape[1]
        self.pool = nn.AdaptiveAvgPool3d(1)
        # Normalise pooled features before the dropout/classification head.
        self.norm = nn.LayerNorm(n_feat)
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(n_feat, n_classes)

    def bottleneck(self, x):
        """Deepest encoder feature map, 512 channels at 1/16 resolution."""
        pyramid = self.encoder(x)
        return pyramid[-1] if isinstance(pyramid, (list, tuple)) else pyramid

    def forward(self, x):
        h = self.pool(self.bottleneck(x)).flatten(1)
        return self.fc(self.drop(self.norm(h)))

    def param_groups(self, encoder_lr, head_lr):
        """Separate encoder and head learning-rate groups."""
        enc = list(self.encoder.parameters())
        head = list(self.norm.parameters()) + list(self.fc.parameters())
        return [{"params": enc, "lr": encoder_lr},
                {"params": head, "lr": head_lr}]


def enable_mc_dropout(model):
    """Eval mode everywhere except dropout, which keeps sampling."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    return model
