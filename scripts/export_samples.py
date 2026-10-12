"""Save one Fashion-MNIST *test* image per class as a PNG, for trying out the UI and API.

    uv run python scripts/export_samples.py [output_dir]      # default: samples/

The test split is never used in training, so these are images the model has not seen.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image
from torchvision.datasets import FashionMNIST

from fashion_mlops.model import CLASS_NAMES


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    test_set = FashionMNIST(root="data", train=False, download=True)

    for index, name in enumerate(CLASS_NAMES):
        position = int((test_set.targets == index).nonzero()[0])
        filename = f"{index}_{name.lower().replace('/', '-').replace(' ', '-')}.png"
        Image.fromarray(test_set.data[position].numpy(), mode="L").save(out_dir / filename)
        print(f"saved {out_dir / filename}")


if __name__ == "__main__":
    main()
