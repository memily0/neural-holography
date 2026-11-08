import math

import torch


def depth_ranking_loss(reconstruction, target, margin=0.2):
    """Require each plane to prefer its own target over dissimilar other targets.

    Correlation rows are reconstruction planes; columns are target planes.
    Targets are detached. Constant-target pairs have no defined correlation
    and are omitted; constant reconstructions remain differentiable through
    the same variance floor used by the training Pearson metric.
    """
    if not math.isfinite(margin) or not 0 <= margin <= 1:
        raise ValueError("margin must be finite and in [0, 1]")
    if reconstruction.ndim != 4 or reconstruction.shape != target.shape or reconstruction.shape[1] < 2:
        raise ValueError("images must have matching [batch, planes>=2, height, width] shapes")
    if not reconstruction.is_floating_point() or not target.is_floating_point():
        raise TypeError("images must be floating point")
    target = target.detach()
    x = reconstruction.flatten(2)
    y = target.flatten(2)
    x = x - x.mean(-1, keepdim=True)
    y = y - y.mean(-1, keepdim=True)
    var_x = x.square().mean(-1)
    var_y = y.square().mean(-1)
    pixels = x.shape[-1]
    covariance = torch.bmm(x, y.transpose(1, 2)) / pixels
    denominator = (var_x.clamp_min(1e-12)[:, :, None] * var_y.clamp_min(1e-12)[:, None, :]).sqrt()
    correlations = (covariance / denominator).clamp(-1, 1)
    target_covariance = torch.bmm(y, y.transpose(1, 2)) / pixels
    target_denominator = (var_y.clamp_min(1e-12)[:, :, None] * var_y.clamp_min(1e-12)[:, None, :]).sqrt()
    target_correlations = (target_covariance / target_denominator).clamp(-1, 1)
    desired_gap = margin * (1 - target_correlations)
    correct = correlations.diagonal(dim1=1, dim2=2)[:, :, None]
    violations = (desired_gap - correct + correlations).clamp_min(0)
    off_diagonal = ~torch.eye(x.shape[1], dtype=torch.bool, device=x.device)
    valid_targets = var_y > 1e-12
    pairs = off_diagonal[None] & valid_targets[:, :, None] & valid_targets[:, None, :]
    return (violations * pairs).sum() / pairs.sum().clamp_min(1)
