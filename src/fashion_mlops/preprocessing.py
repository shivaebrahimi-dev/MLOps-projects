"""Image preprocessing shared by training and inference.

Both paths end in ``to_model_input``, so an image sent to the API is scaled and
normalised exactly like the images the model was trained on.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

IMAGE_SIZE = 28
# Mean and standard deviation of the Fashion-MNIST training pixels, on a 0-1 scale.
FASHION_MNIST_MEAN = 0.2860
FASHION_MNIST_STD = 0.3530


def to_model_input(pixels: torch.Tensor) -> torch.Tensor:
    """Convert 0-255 pixel values of shape (..., 28, 28) to normalised (..., 1, 28, 28)."""
    scaled = pixels.float().div(255.0)
    return scaled.sub(FASHION_MNIST_MEAN).div(FASHION_MNIST_STD).unsqueeze(-3)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Turn any PIL image into a model-ready batch of shape (1, 1, 28, 28)."""
    gray = image.convert("L")
    if gray.size != (IMAGE_SIZE, IMAGE_SIZE):
        gray = gray.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    pixels = torch.from_numpy(np.array(gray, dtype=np.uint8))
    return to_model_input(pixels).unsqueeze(0)
