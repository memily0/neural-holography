"""Pure learning-rate schedules indexed by the absolute optimizer update."""

import math


def cosine_lr_multiplier(step, total_steps, warmup_steps=0, min_multiplier=0.1):
    """Return a multiplier for the 1-based update ``step``.

    Without warmup, update 1 uses the base learning rate. With warmup, the
    first updates use step/warmup_steps and reach the base rate at the final
    warmup update. Cosine decay then reaches min_multiplier at total_steps.
    The floor applies to the decay, not to warmup. A one-update run uses 1.
    Always multiply the configured base LR, never the previous update's LR.
    """
    if any(not isinstance(value, int) or isinstance(value, bool) for value in (step, total_steps, warmup_steps)):
        raise TypeError("step, total_steps and warmup_steps must be integers")
    if total_steps < 1 or not 1 <= step <= total_steps:
        raise ValueError("step must be in [1, total_steps] and total_steps positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    if not math.isfinite(min_multiplier) or not 0 <= min_multiplier <= 1:
        raise ValueError("min_multiplier must be finite and in [0, 1]")
    if warmup_steps and step <= warmup_steps:
        return step / warmup_steps
    if total_steps == 1:
        return 1.0
    start = max(1, warmup_steps)
    progress = (step - start) / (total_steps - start)
    return min_multiplier + 0.5 * (1 - min_multiplier) * (1 + math.cos(math.pi * progress))
