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


def to_fashion_mnist_layout(gray: Image.Image) -> Image.Image:
    """Rearrange an ordinary grayscale picture into the layout Fashion-MNIST uses.

    The dataset was made from product photos by negating them, trimming the empty
    margin and centring the item in a 28x28 frame, so every training image is a
    light item filling a black square. A normal photo (dark item, white background,
    wide margins) looks nothing like that to the model unless the same steps are applied.
    """
    pixels = np.asarray(gray, dtype=np.float32)
    border = np.concatenate([pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]])
    if np.median(border) > 127:  # light background: negate, as the dataset did
        pixels = 255.0 - pixels
        border = 255.0 - border

    # Push the background to pure black and stretch the rest back to the full range.
    background = float(np.median(border))
    pixels = np.clip((pixels - background) * 255.0 / max(255.0 - background, 1.0), 0, 255)

    # Trim the empty margin so the item fills the frame.
    rows, cols = np.nonzero(pixels > 25)
    if rows.size:
        pixels = pixels[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

    item = Image.fromarray(pixels.astype(np.uint8), mode="L")
    scale = IMAGE_SIZE / max(item.size)
    size = (max(1, round(item.width * scale)), max(1, round(item.height * scale)))
    item = item.resize(size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
    canvas.paste(item, ((IMAGE_SIZE - item.width) // 2, (IMAGE_SIZE - item.height) // 2))
    return canvas


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Turn any PIL image into a model-ready batch of shape (1, 1, 28, 28).

    A 28x28 image is taken to be in Fashion-MNIST format already and only goes through
    ``to_model_input``, exactly like the training data. Anything else is first brought
    into that format by ``to_fashion_mnist_layout``.
    """
    if image.mode in ("RGBA", "LA", "P"):
        # Flatten transparency onto white, the usual background of a product photo.
        rgba = image.convert("RGBA")
        flattened = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(flattened, rgba)
    gray = image.convert("L")
    if gray.size != (IMAGE_SIZE, IMAGE_SIZE):
        gray = to_fashion_mnist_layout(gray)
    pixels = torch.from_numpy(np.array(gray, dtype=np.uint8))
    return to_model_input(pixels).unsqueeze(0)
