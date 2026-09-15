"""The classifier built on the CT-FM encoder.

CT-FM is a SegResNet pretrained with contrastive self-supervised learning on
148,000 CT scans from the Imaging Data Commons. We keep its encoder, drop the
decoder, and put a small classification head on the bottleneck.

Three details worth recording, because each one cost time to find and each one
would have produced a result that looked fine and meant nothing.

The published checkpoint does not load into MONAI's SegResNetDS. The parameter
names differ, `load_state_dict(strict=False)` matches zero tensors, and training
proceeds on a randomly initialised network while the code looks correct. The
`lighter_zoo` loader is the one that works.

CT-FM downsamples by 16, so input side lengths must be divisible by 16. A 28^3
volume raises an error, which is why the 64^3 release of the dataset is used.

The encoder returns a five-level feature pyramid, and the full SegResNet output
is a 2-channel volume at input resolution. Pooling that output gives a linear
layer exactly two features to classify from. We take the bottleneck instead,
512 channels at 4^3, which is the representation the self-supervised pretraining
actually shaped.
"""

import torch
import torch.nn as nn
from lighter_zoo import SegResNet

HF_ID = "project-lighter/ct_fm_feature_extractor"
CACHE = "data/hf"


def build_encoder(pretrained=True, cache_dir=CACHE):
    """CT-FM encoder, with or without the pretrained weights.

    The from-scratch arm reinitialises the loaded model rather than constructing
    a second network. Building one from a guessed config gave an architecture
    with 19.9M parameters against the real 87.2M, which would have made the
    ablation compare two different models and report the gap as an effect of
    pretraining.
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
        # The pooled CT-FM features have a standard deviation around 0.03. Fed
        # straight into a linear layer the head learns nothing useful and the
        # training loss sits an order of magnitude above chance. LayerNorm puts
        # them on a scale the head can work with.
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
        """Lower learning rate for the pretrained encoder than for the new head.

        Fine-tuning 87M pretrained parameters at the head's learning rate walks
        the representation away from what the self-supervised training produced,
        which is the thing the ablation is supposed to measure.
        """
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
