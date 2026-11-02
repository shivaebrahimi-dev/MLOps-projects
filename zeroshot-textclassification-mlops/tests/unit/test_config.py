import json
import logging

import pytest

from app.utils.config import DEFAULT_BUCKET, PROJECT_ROOT, parse_settings
from app.utils.errors import ConfigurationError
from app.utils.logging_config import REDACTED, JsonFormatter, log_event

REQUIRED = {"MINIO_ACCESS_KEY": "access-value", "MINIO_SECRET_KEY": "secret-value"}


def test_defaults_are_applied():
    settings = parse_settings(REQUIRED)

    assert settings.minio_endpoint == "localhost:9000"
    assert settings.minio_bucket == DEFAULT_BUCKET == "zeroshot-learning-output"
    assert settings.minio_secure is False
    assert settings.max_input_chars == 10000
    assert settings.hf_token is None
    assert settings.model_dir == PROJECT_ROOT / "models"


def test_values_come_from_the_environment():
    settings = parse_settings(
        {
            **REQUIRED,
            "MINIO_BUCKET": "other-bucket",
            "MINIO_SECURE": "true",
            "MAX_INPUT_CHARS": "50",
        }
    )

    assert settings.minio_bucket == "other-bucket"
    assert settings.minio_secure is True
    assert settings.max_input_chars == 50


@pytest.mark.parametrize("missing", ["MINIO_ACCESS_KEY", "MINIO_SECRET_KEY"])
def test_missing_required_value_names_the_variable(missing):
    env = {name: value for name, value in REQUIRED.items() if name != missing}

    with pytest.raises(ConfigurationError, match=missing):
        parse_settings(env)


@pytest.mark.parametrize(
    ("name", "value"),
    [("MINIO_SECURE", "maybe"), ("MAX_INPUT_CHARS", "many"), ("MAX_INPUT_CHARS", "0")],
)
def test_invalid_values_are_rejected_without_echoing_them(name, value):
    with pytest.raises(ConfigurationError, match=name) as error:
        parse_settings({**REQUIRED, name: value})

    assert value not in str(error.value)


def test_secrets_are_not_in_the_settings_text():
    text = repr(parse_settings({**REQUIRED, "HF_TOKEN": "hf_tokenvalue123"}))

    assert "secret-value" not in text
    assert "access-value" not in text
    assert "hf_tokenvalue123" not in text


def _formatted_entry(caplog, **fields):
    logger = logging.getLogger("test")
    with caplog.at_level(logging.INFO, logger="test"):
        log_event(logger, "event_name", **fields)
    return json.loads(JsonFormatter().format(caplog.records[-1]))


def test_logs_are_json_with_metadata(caplog):
    entry = _formatted_entry(caplog, model_id="org/name", duration_ms=12)

    assert entry["event"] == "event_name"
    assert entry["level"] == "INFO"
    assert entry["model_id"] == "org/name"
    assert entry["duration_ms"] == 12


def test_logs_redact_credentials(caplog):
    entry = _formatted_entry(
        caplog,
        minio_secret_key="secret-value",
        hf_token="hf_tokenvalue123",
        detail="request failed with hf_tokenvalue123",
    )

    assert entry["minio_secret_key"] == REDACTED
    assert entry["hf_token"] == REDACTED
    assert "hf_tokenvalue123" not in entry["detail"]


def test_logs_omit_input_text(caplog):
    entry = _formatted_entry(caplog, input_text="private text", text="private text")

    assert "input_text" not in entry
    assert "text" not in entry
