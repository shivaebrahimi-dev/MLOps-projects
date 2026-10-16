import io
from pathlib import Path

import httpx
import pytest
from PIL import Image

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).parent.parent / "src" / "fashion_mlops" / "ui" / "app.py")

PREDICTION = {
    "predicted_class": "Sneaker",
    "class_index": 7,
    "confidence": 0.93,
    "probabilities": {"Sneaker": 0.93, "Sandal": 0.07},
    "model_object_key": "model/run-1/model.pt",
}


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (28, 28), color=200).save(buffer, format="PNG")
    return buffer.getvalue()


def fake_api(monkeypatch, health=None, predict=None):
    """Replace the UI's HTTP calls. ``None`` for health simulates an unreachable API."""

    def get(url, **kwargs):
        if health is None:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json=health, request=httpx.Request("GET", url))

    def post(url, **kwargs):
        status, body = predict
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(httpx, "post", post)


def upload_and_predict(app: AppTest) -> AppTest:
    app.file_uploader[0].upload("item.png", png_bytes(), "image/png")
    app.run()
    return app.button[0].click().run()


def test_unreachable_api_shows_message(monkeypatch):
    fake_api(monkeypatch, health=None)
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert any("could not be reached" in e.value for e in app.error)


def test_missing_model_shows_helpful_message(monkeypatch):
    fake_api(monkeypatch, health={"status": "ok", "model_available": False})
    app = AppTest.from_file(APP).run()
    assert not app.exception
    assert any("No trained model is available yet" in w.value for w in app.warning)


def test_prediction_is_displayed(monkeypatch):
    health = {"status": "ok", "model_available": True, "model_object_key": "model/run-1/model.pt"}
    fake_api(monkeypatch, health=health, predict=(200, PREDICTION))
    app = upload_and_predict(AppTest.from_file(APP).run())
    assert not app.exception
    assert [(m.label, m.value) for m in app.metric] == [
        ("Predicted class", "Sneaker"),
        ("Confidence", "93.0%"),
    ]


def test_model_loading_error_is_displayed(monkeypatch):
    health = {"status": "ok", "model_available": False}
    fake_api(monkeypatch, health=health, predict=(503, {"detail": "No trained model."}))
    app = upload_and_predict(AppTest.from_file(APP).run())
    assert not app.exception
    assert any("not available yet" in e.value for e in app.error)
    assert not app.metric
