import torch
from torch.nn import functional as F


def _check_images(x, y):
    if x.ndim != 4 or x.shape != y.shape:
        raise ValueError("images must have matching [B,C,H,W] shapes")
    if not x.is_floating_point() or not y.is_floating_point():
        raise TypeError("metrics expect floating point images normalized to [0,1]")


def pearson(x, y, return_valid=False, eps=1e-12):
    """Return finite [B,C] values for training, plus optional validity flags.

    Evaluation must mark entries with ``valid=False`` as undefined. The
    denominator floor keeps gradients finite when a prediction is constant;
    it does not silently count such a prediction as a valid correlation.
    """
    _check_images(x, y)
    x = x - x.mean(dim=(-2, -1), keepdim=True)
    y = y - y.mean(dim=(-2, -1), keepdim=True)
    var_x = x.square().mean(dim=(-2, -1))
    var_y = y.square().mean(dim=(-2, -1))
    covariance = (x * y).mean(dim=(-2, -1))
    denominator = (var_x.clamp_min(eps) * var_y.clamp_min(eps)).sqrt()
    values = (covariance / denominator).clamp(-1, 1)
    valid = (var_x > eps) & (var_y > eps)
    return (values, valid) if return_valid else values


def _ssim_values(mean_x, mean_y, second_x, second_y, cross, data_range):
    var_x = second_x - mean_x.square()
    var_y = second_y - mean_y.square()
    covariance = cross - mean_x * mean_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mean_x * mean_y + c1) * (2 * covariance + c2)
    denominator = (mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2)
    return numerator / denominator


def ssim(x, y, window_size=11, sigma=1.5, data_range=1.0):
    """Gaussian SSIM, population covariance, valid window, output [B,C]."""
    _check_images(x, y)
    if window_size < 1 or window_size % 2 != 1 or min(x.shape[-2:]) < window_size:
        raise ValueError("window_size must be odd and fit inside both image dimensions")
    if sigma <= 0 or data_range <= 0:
        raise ValueError("sigma and data_range must be positive")
    coordinates = torch.arange(window_size, device=x.device, dtype=x.dtype) - window_size // 2
    kernel = torch.exp(-coordinates.square() / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    moments = torch.cat((x, y, x.square(), y.square(), x * y), dim=1)
    channels = moments.shape[1]
    horizontal = kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1)
    moments = F.conv2d(moments, horizontal, groups=channels)
    moments = F.conv2d(moments, vertical, groups=channels)
    values = _ssim_values(*moments.chunk(5, dim=1), data_range)
    return values.mean(dim=(-2, -1))


def global_ssim(x, y, data_range=1.0):
    """Whole-image SSIM from the task slide, with population covariance."""
    _check_images(x, y)
    moments = [value.mean(dim=(-2, -1)) for value in (x, y, x.square(), y.square(), x * y)]
    return _ssim_values(*moments, data_range)
