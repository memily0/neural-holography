"""Source-feature calibration and training-only complex-field consistency."""

import torch


def balanced_source_features(source_fields, target):
    """Calibrate each complex source feature by a reference target RMS.

    For unchanged target moduli, unitary transport preserves target RMS, so
    this yields unit-RMS features. With a learned envelope correction it is
    still the original target's reference scale, not automatic normalization
    of the corrected feature energy. No input or physical mask is modified.
    An all-zero target produces all-zero features and finite zero gradients.
    """
    if source_fields.ndim != 4 or not source_fields.is_complex() or min(source_fields.shape) < 1:
        raise ValueError("source_fields must be nonempty complex [batch, planes, height, width]")
    if target.shape != source_fields.shape or not target.is_floating_point():
        raise ValueError("target must be a floating amplitude tensor matching source_fields")
    target_energy = target.to(source_fields.real.dtype).square().mean((-2, -1), keepdim=True)
    valid = target_energy > 0
    # Clamp before sqrt to keep zero-target gradients finite. Genuine positive
    # uint8 targets on the task grid have RMS well above this numerical guard.
    target_rms = target_energy.clamp_min(1e-16).sqrt()
    return torch.where(valid, source_fields / target_rms, torch.zeros_like(source_fields))


def latent_source_consistency(source_fields, phase, target):
    """Mean complex error to the common unit-amplitude generated DOE field.

    For fixed target moduli, source_fields = P_j* [target_j * exp(i theta_j)]
    has the target RMS because P is unitary. For learned corrected envelopes,
    the original target RMS remains the reference. Zero loss then also
    requires preservation of that original energy: source_j = RMS(target_j)
    * exp(i phase). This is an intentional energy-and-consistency constraint,
    not a claim that arbitrary corrected features already have unit RMS.
    Zero targets impose no constraint and are excluded from the mean.

    Both source fields and generated phase receive gradients. This objective
    trains shared network weights; it does not fit or refine an input mask.
    """
    normalized = balanced_source_features(source_fields, target)
    if phase.ndim == 3:
        phase = phase[:, None]
    expected = (source_fields.shape[0], 1, *source_fields.shape[-2:])
    if phase.shape != expected or not phase.is_floating_point():
        raise ValueError("phase must be floating [batch, 1, height, width] or [batch, height, width]")
    dtype = source_fields.real.dtype
    target_energy = target.to(dtype).square().mean((-2, -1), keepdim=True)
    valid = target_energy > 0
    generated = torch.exp(1j * phase.to(dtype))
    difference = normalized - generated
    errors = (difference.real.square() + difference.imag.square()).mean((-2, -1), keepdim=True)
    return (errors * valid).sum() / valid.sum().clamp_min(1)
