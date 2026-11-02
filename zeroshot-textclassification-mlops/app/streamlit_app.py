"""Home page of the zero-shot text classification platform."""

import sys
from pathlib import Path

# `streamlit run` only adds this folder to the import path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.ui_state import get_services

st.set_page_config(page_title="Zero-Shot Text Classification")
st.title("Zero-Shot Text Classification")
st.write(
    "Classify text into categories you choose, without training a model. "
    "Every successful result is saved to MinIO for later analysis."
)

services = get_services()
ready = len(services.models.ready_models())
total = len(services.models.configs())
st.metric("Models ready", f"{ready} of {total}")

st.markdown(
    "- **Admin**: download and delete models, clear the cache, and manage label sets.\n"
    "- **Classify Text**: submit text and view ranked results."
)
st.info("This application has no login. Use it only for local development or trusted environments.")
