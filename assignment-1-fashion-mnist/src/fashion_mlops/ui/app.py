"""Streamlit front end for the Fashion-MNIST classifier.

    streamlit run src/fashion_mlops/ui/app.py

This app does no inference itself: every prediction is a call to the FastAPI
service at ``API_URL``. It deliberately imports nothing from the rest of the
package so its container needs neither PyTorch nor the model code.
"""

from __future__ import annotations

import io
import os

import httpx
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TRAIN_HINT = "Train a model first with `docker compose run --rm trainer`, then try again."

st.set_page_config(page_title="Fashion-MNIST classifier", page_icon="👕", layout="centered")


def fetch_health() -> dict | None:
    """Return the API's /health body, or None if the API cannot be reached."""
    try:
        response = httpx.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        return None


def request_prediction(name: str, data: bytes, content_type: str) -> tuple[dict | None, str | None]:
    """Call /predict. Returns (prediction, None) on success or (None, message) on failure."""
    try:
        response = httpx.post(
            f"{API_URL}/predict", files={"file": (name, data, content_type)}, timeout=30
        )
    except httpx.HTTPError:
        return None, "The prediction service could not be reached. Is the `api` container running?"

    if response.status_code == 200:
        return response.json(), None
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    if response.status_code == 503:
        return None, f"The model is not available yet. {detail} {TRAIN_HINT}"
    return None, f"The image was rejected: {detail}"


def show_service_status(health: dict | None) -> None:
    with st.sidebar:
        st.header("Service status")
        if health is None:
            st.error("Prediction API: unreachable")
            st.caption(f"Tried {API_URL}")
        elif health["model_available"]:
            st.success("Prediction API: ready")
            st.caption("Serving model")
            st.code(health["model_object_key"], language=None)
        else:
            st.warning("Prediction API: running, no model yet")
            st.caption(TRAIN_HINT)

        st.header("Classes")
        st.caption(
            "T-shirt/top · Trouser · Pullover · Dress · Coat · "
            "Sandal · Shirt · Sneaker · Bag · Ankle boot"
        )


def show_prediction(prediction: dict) -> None:
    st.metric("Predicted class", prediction["predicted_class"])
    st.metric("Confidence", f"{prediction['confidence']:.1%}")


def show_probabilities(prediction: dict) -> None:
    table = (
        pd.DataFrame(
            {"class": list(prediction["probabilities"]),
             "probability": list(prediction["probabilities"].values())}
        )
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )
    st.subheader("Class probabilities")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "class": "Class",
            "probability": st.column_config.ProgressColumn(
                "Probability", format="percent", min_value=0.0, max_value=1.0
            ),
        },
    )


def main() -> None:
    st.title("Fashion-MNIST classifier")
    st.write(
        "Upload a picture of a clothing item and a LeNet-5 model will predict which of "
        "the ten Fashion-MNIST classes it belongs to."
    )

    health = fetch_health()
    show_service_status(health)
    if health is None:
        st.error(
            "The prediction service could not be reached, so predictions are unavailable "
            "right now. Start it with `docker compose up -d api`."
        )
    elif not health["model_available"]:
        st.warning(f"No trained model is available yet. {TRAIN_HINT}")

    uploaded = st.file_uploader(
        "Choose one image",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        accept_multiple_files=False,
        help="Works best with Fashion-MNIST-style images: a light item on a black background.",
    )
    if uploaded is None:
        st.info("Sample test images are in the repository's `samples/` folder.")
        return

    data = uploaded.getvalue()
    preview_column, result_column = st.columns(2, gap="large")

    with preview_column:
        try:
            image = Image.open(io.BytesIO(data))
            # Tiny images are enlarged without smoothing so the individual pixels stay visible.
            if max(image.size) < 128:
                image = image.resize((224, 224), Image.Resampling.NEAREST)
            st.image(image, caption=uploaded.name, width=224)
        except (UnidentifiedImageError, OSError):
            st.error("This file could not be displayed as an image.")

    with result_column:
        clicked = st.button("Predict", type="primary", width="stretch")
        if clicked:
            with st.spinner("Asking the model..."):
                prediction, error = request_prediction(uploaded.name, data, uploaded.type)
            # Kept in session state so the result survives Streamlit's reruns.
            st.session_state["result"] = (uploaded.file_id, prediction, error)

        file_id, prediction, error = st.session_state.get("result", (None, None, None))
        if file_id != uploaded.file_id:
            prediction, error = None, None
        if error:
            st.error(error)
        elif prediction:
            show_prediction(prediction)

    if prediction:
        show_probabilities(prediction)
        st.caption(f"Model: `{prediction['model_object_key']}`")


main()
