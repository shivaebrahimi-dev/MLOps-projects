"""User page for classifying text and viewing ranked results."""

import sys
import uuid
from pathlib import Path

# `streamlit run` only adds the app folder to the import path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.services.classifier import ClassificationResult
from app.ui_state import Services, get_services
from app.utils.errors import StorageError, ZeroShotError


def show_result(result: ClassificationResult) -> None:
    st.subheader("Result")
    st.write("**Input text**")
    st.text(result.input_text)
    predicted, confidence = st.columns(2)
    predicted.metric("Predicted class", result.predicted_class)
    confidence.metric("Confidence", f"{result.confidence:.2f}")

    st.write("**Ranked predictions**")
    st.table(
        [
            {"Rank": rank, "Label": item.label, "Score": f"{item.score:.2f}"}
            for rank, item in enumerate(result.ranked_predictions, start=1)
        ]
    )

    st.write("**Details**")
    st.text(
        f"Model: {result.model_name} ({result.model_id})\n"
        f"Revision: {result.model_revision or 'unavailable'}\n"
        f"Timestamp: {result.timestamp}\n"
        f"Request ID: {result.request_id}"
    )


def save_result(services: Services, result: ClassificationResult, session_id: str) -> None:
    try:
        stored = services.storage.save(result, session_id=session_id)
    except StorageError as error:
        st.error(f"{error.user_message} Request ID: {result.request_id}")
    else:
        st.success(f"Saved to MinIO: {stored.bucket}/{stored.object_path}")


st.set_page_config(page_title="Classify Text")
st.title("Classify Text")

services = get_services()
session_id = st.session_state.setdefault("session_id", str(uuid.uuid4()))

try:
    ready_models = services.models.ready_models()
    label_sets = services.labels.label_sets()
    default_label_set = services.labels.default_name()
except ZeroShotError as error:
    st.error(error.user_message)
    st.stop()

if not ready_models:
    st.warning("No model is ready yet. Download a model on the Admin page, then come back here.")
    st.stop()

names = [label_set.name for label_set in label_sets]
default_index = next(
    (i for i, name in enumerate(names) if name.casefold() == default_label_set.casefold()), 0
)
limit = services.settings.max_input_chars

with st.form("classify"):
    text = st.text_area("Text to classify", height=160)
    st.caption(f"Up to {limit:,} characters.")
    selected_model = st.selectbox(
        "Model", ready_models, format_func=lambda model: model.config.name
    )
    selected_set = st.selectbox(
        "Label set",
        label_sets,
        index=default_index,
        format_func=lambda label_set: f"{label_set.name} ({', '.join(label_set.labels)})",
    )
    submitted = st.form_submit_button("Classify")

if submitted:
    try:
        with st.spinner("Classifying. The first run with a model takes longer while it loads."):
            classification = services.classifier.classify(
                text, selected_model.config.id, selected_set.labels
            )
    except ZeroShotError as error:
        st.error(error.user_message)
    else:
        show_result(classification)
        save_result(services, classification, session_id)
