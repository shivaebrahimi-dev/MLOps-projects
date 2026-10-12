"""LeNet-5-style convolutional network for 28x28 grayscale images."""

from __future__ import annotations

import torch
from torch import nn

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


class LeNet5(nn.Module):
    """LeNet-5 adapted to 28x28 inputs.

    The original network expects 32x32 images; padding the first convolution by 2
    gives the same feature-map sizes for 28x28 inputs. ReLU and max pooling replace
    the original tanh and average pooling, as is usual in modern variants.
    """

    def __init__(self, num_classes: int = len(CLASS_NAMES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 1x28x28 -> 6x28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 6x14x14
            nn.Conv2d(6, 16, kernel_size=5),  # -> 16x10x10
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 16x5x5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw class scores (logits) of shape (batch, num_classes)."""
        return self.classifier(self.features(x))
