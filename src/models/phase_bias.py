import torch
from torch import nn


class SharedPhaseBias(nn.Module):
    """Add a learned dataset-wide phase prior to a feed-forward generator.

    The zero-initialized bias is an ordinary shared network parameter. It is
    trained jointly across batches, never fitted or updated for an inference
    input. No analytic lens, propagation inverse or classical mask initializes
    it. The final phase is wrapped to [-pi, pi] in one forward pass.
    """

    def __init__(self, base_generator):
        super().__init__()
        self.base_generator = base_generator
        self.phase_bias = nn.Parameter(torch.zeros(1, 1, 128, 128))

    def forward(self, images, return_diagnostics=False):
        result = self.base_generator(images, return_diagnostics=return_diagnostics)
        if return_diagnostics:
            phase, diagnostics = result
        else:
            phase = result
        if phase.ndim != 4 or phase.shape[1:] != (1, 128, 128):
            raise ValueError("base generator must return phase shaped [batch, 1, 128, 128]")
        shifted = phase + self.phase_bias.to(phase.dtype)
        wrapped = torch.atan2(shifted.sin(), shifted.cos())
        if return_diagnostics:
            diagnostics = dict(diagnostics)
            diagnostics["shared_phase_bias_rms"] = self.phase_bias.detach().square().mean().sqrt()
            return wrapped, diagnostics
        return wrapped
