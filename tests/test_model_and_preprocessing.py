import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from fashion_mlops.config import TrainConfig
from fashion_mlops.data import split_indices
from fashion_mlops.model import CLASS_NAMES, LeNet5
from fashion_mlops.preprocessing import (
    FASHION_MNIST_MEAN,
    FASHION_MNIST_STD,
    preprocess_image,
    to_model_input,
)
from fashion_mlops.train import train_model


def test_model_output_shape():
    logits = LeNet5()(torch.zeros(4, 1, 28, 28))
    assert logits.shape == (4, len(CLASS_NAMES))


def test_to_model_input_normalises_and_adds_channel():
    pixels = torch.tensor([[0, 255]], dtype=torch.uint8).expand(28, 2).repeat(1, 14)
    out = to_model_input(pixels)
    assert out.shape == (1, 28, 28)
    assert torch.isclose(out.min(), torch.tensor(-FASHION_MNIST_MEAN / FASHION_MNIST_STD))
    assert torch.isclose(out.max(), torch.tensor((1 - FASHION_MNIST_MEAN) / FASHION_MNIST_STD))


def test_preprocess_image_handles_colour_and_size():
    image = Image.new("RGB", (100, 60), color=(255, 255, 255))
    out = preprocess_image(image)
    assert out.shape == (1, 1, 28, 28)
    assert out.dtype == torch.float32


def test_preprocess_matches_training_path():
    """A 28x28 grayscale image must come out identical to the training transform."""
    pixels = torch.randint(0, 256, (28, 28), dtype=torch.uint8)
    image = Image.fromarray(pixels.numpy(), mode="L")
    assert torch.equal(preprocess_image(image)[0], to_model_input(pixels))


def test_split_is_disjoint_complete_and_reproducible():
    train_idx, val_idx = split_indices(1000, 0.2, seed=7)
    assert len(val_idx) == 200 and len(train_idx) == 800
    assert set(train_idx.tolist()).isdisjoint(val_idx.tolist())
    assert sorted(train_idx.tolist() + val_idx.tolist()) == list(range(1000))
    again_train, again_val = split_indices(1000, 0.2, seed=7)
    assert torch.equal(train_idx, again_train) and torch.equal(val_idx, again_val)
    assert not torch.equal(val_idx, split_indices(1000, 0.2, seed=8)[1])


def test_train_model_runs_and_reports_metrics():
    torch.manual_seed(0)
    data = TensorDataset(torch.randn(64, 1, 28, 28), torch.randint(0, 10, (64,)))
    loader = DataLoader(data, batch_size=16)
    seen = []
    config = TrainConfig(epochs=2, loss="nll", optimizer="sgd")

    result = train_model(config, loader, loader, on_epoch_end=lambda e, m: seen.append(e))

    assert seen == [1, 2]
    assert result.epochs_run == 2
    assert set(result.final_metrics) == {"train_loss", "val_loss", "val_accuracy"}
    assert 0.0 <= result.final_metrics["val_accuracy"] <= 1.0
