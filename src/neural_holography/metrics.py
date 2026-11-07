import torch
import torch.nn.functional as F


def pearson_per_plane(p, t):
    a = p - p.mean(dim=(-2, -1), keepdim=True)
    b = t - t.mean(dim=(-2, -1), keepdim=True)
    numerator = (a * b).sum((-2, -1))
    den = (a.square().sum((-2, -1)) * b.square().sum((-2, -1))).sqrt()
    return torch.where(den > 1e-8, numerator / den.clamp_min(1e-8), torch.zeros_like(den))


def global_ssim_per_plane(p, t):
    mp, mt = p.mean((-2, -1)), t.mean((-2, -1))
    dp, dt = p - mp[..., None, None], t - mt[..., None, None]
    vp, vt = dp.square().mean((-2, -1)), dt.square().mean((-2, -1))
    cov = (dp * dt).mean((-2, -1))
    c1, c2 = .01**2, .03**2
    return (2 * mp * mt + c1) * (2 * cov + c2) / ((mp.square() + mt.square() + c1) * (vp + vt + c2))


def local_ssim_per_plane(p, t, window=7):
    b, c, h, w = p.shape
    a = p.reshape(b * c, 1, h, w)
    q = t.reshape(b * c, 1, h, w)

    def avg(x):
        return F.avg_pool2d(F.pad(x, [window // 2] * 4, mode="reflect"), window, stride=1)

    ma, mb = avg(a), avg(q)
    va = (avg(a * a) - ma.square()).clamp_min(0)
    vb = (avg(q * q) - mb.square()).clamp_min(0)
    cov = avg(a * q) - ma * mb
    c1, c2 = .01**2, .03**2
    s = (2 * ma * mb + c1) * (2 * cov + c2) / ((ma.square() + mb.square() + c1) * (va + vb + c2))
    return s.mean((-2, -1)).reshape(b, c)


def calc_metrics(pred, target):
    metrics = {
        "cc": pearson_per_plane(pred, target),
        "local": local_ssim_per_plane(pred, target),
        "global": global_ssim_per_plane(pred, target),
    }
    return {key + "_mean": float(value.mean()) for key, value in metrics.items()} | {
        key + "_planes": [float(v) for v in value.mean(0)] for key, value in metrics.items()
    }
