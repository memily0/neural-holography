# Neural Holography — starter files

This starter is meant to be **extracted into the root of an existing GitHub repository**.
It does not overwrite your main `README.md` or `.gitignore`.

## Physics module

- `src/physics/fresnel.py`: one-FFT Fresnel model and 4-plane reconstruction.
- `tests/test_fresnel.py`: shapes, finite values, autograd, energy conservation.

## Setup (Mac/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-physics.txt
python -m pytest tests/test_fresnel.py -q
```

## Interfaces agreed for the team

```python
from src.physics.fresnel import fresnel_propagate, reconstruct_four

# phase shape: (B, 1, 128, 128), float32, radians [-pi, pi]
reconstruction = reconstruct_four(phase)
# result shape: (B, 4, 128, 128), amplitude |U|, float32
```

Distances are in **meters**: `(0.0241, 0.0253, 0.0265, 0.0277)`.
Wavelength = 532e-9 meters. Input SLM pixel pitch = 10e-6 meters.

**Important:** physical output pixel pitch of one-FFT Fresnel propagation
changes with distance: `wavelength * distance / (N * input_pixel_pitch)`.
Confirm output-grid interpretation and normalization with the organizers
before relying on absolute quality scores.

Data and `outputs/` are starter directories, not training datasets.
The physics module returns **amplitude** `abs(U)` rather than intensity `abs(U)**2`
following the task's description of saving magnitudes.
