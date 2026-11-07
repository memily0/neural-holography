import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from neural_holography.config import DEFAULT_CHECKPOINT
from neural_holography.inference import predict, select_device
from neural_holography.model import load_model


def main():
    parser = argparse.ArgumentParser(description="Four grayscale PNGs to one phase DOE")
    parser.add_argument("--input", nargs=4, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--reconstruct-dir", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    try:
        device = select_device(args.device)
        model, _ = load_model(args.checkpoint, device)
        predict(args.input, args.output, model, device, args.reconstruct_dir)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
