"""A learned residual real envelope for complex target-coordinate features."""

import torch
from torch import nn
from torch.nn import functional as F

from src.models.transport import TransportEncoder


class ZeroEnvelopeHead(nn.Conv2d):
    """Zero initialization without consuming RNG or changing other layers."""

    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class TargetEnvelopeEncoder(TransportEncoder):
    """Keep the phase U-Net and add four real residual-envelope predictions.

    Both heads use the same decoded features. Existing phase-network state
    keys stay unchanged; only envelope.weight and envelope.bias are new.
    The delta is signed and unconstrained: images + delta describes learned
    complex latent features, not a nonnegative physical target amplitude.
    The final phase mask still comes from the separate learned source decoder.
    """

    def __init__(self, in_channels, out_channels, width):
        if out_channels != 4:
            raise ValueError("The target envelope requires four output planes")
        super().__init__(in_channels, out_channels, width)
        self.envelope = ZeroEnvelopeHead(width, out_channels, 1)

    def forward(self, images):
        first = self.encoder1(images)
        second = self.encoder2(F.avg_pool2d(first, 2))
        latent = self.bottleneck(F.avg_pool2d(second, 2))
        decoded = self.decoder2(torch.cat((self.up2(latent), second), dim=1))
        decoded = self.decoder1(torch.cat((self.up1(decoded), first), dim=1))
        return self.output(decoded), self.envelope(decoded)
