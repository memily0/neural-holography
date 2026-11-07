from pathlib import Path

SIZE = 128
DISTANCES = (0.0241, 0.0253, 0.0265, 0.0277)
WAVELENGTH = 532e-9
PIXEL_PITCH = 10e-6
ARCHITECTURE = "smp.UnetPlusPlus/resnet34/seed8"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "checkpoints" / "best_model.pt"
