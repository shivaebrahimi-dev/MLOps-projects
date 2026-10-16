from pathlib import Path

import pytest
from pydantic import ValidationError

from fashion_mlops.config import TrainConfig, load_train_config

EXAMPLE_CONFIG = Path(__file__).parent.parent / "configs" / "train.toml"


def write_toml(tmp_path, body: str) -> Path:
    path = tmp_path / "train.toml"
    path.write_text("[training]\n" + body)
    return path


def test_defaults_are_valid():
    config = load_train_config(env={})
    assert config == TrainConfig()


def test_example_config_loads():
    config = load_train_config(EXAMPLE_CONFIG, env={})
    assert config.epochs == 5
    assert config.optimizer == "adam"


def test_toml_values_are_used(tmp_path):
    path = write_toml(tmp_path, 'epochs = 3\noptimizer = "sgd"\n')
    config = load_train_config(path, env={})
    assert config.epochs == 3
    assert config.optimizer == "sgd"


def test_env_overrides_toml(tmp_path):
    path = write_toml(tmp_path, "epochs = 3\nlearning_rate = 0.1\n")
    config = load_train_config(path, env={"TRAIN_EPOCHS": "7"})
    assert config.epochs == 7
    assert config.learning_rate == 0.1


def test_cli_overrides_env_and_toml(tmp_path):
    path = write_toml(tmp_path, "epochs = 3\n")
    config = load_train_config(path, overrides={"epochs": 9}, env={"TRAIN_EPOCHS": "7"})
    assert config.epochs == 9


def test_unset_cli_flags_do_not_override(tmp_path):
    path = write_toml(tmp_path, "epochs = 3\n")
    config = load_train_config(path, overrides={"epochs": None}, env={})
    assert config.epochs == 3


@pytest.mark.parametrize(
    "values",
    [
        {"epochs": 0},
        {"learning_rate": -1},
        {"val_split": 1.5},
        {"optimizer": "rmsprop"},
        {"loss": "mse"},
    ],
)
def test_invalid_values_are_rejected(values):
    with pytest.raises(ValidationError):
        load_train_config(overrides=values, env={})


def test_unknown_toml_key_is_rejected(tmp_path):
    path = write_toml(tmp_path, "epocs = 3\n")
    with pytest.raises(ValidationError):
        load_train_config(path, env={})
