"""Administrator page for managing models and candidate labels."""

import sys
from pathlib import Path

# `streamlit run` only adds the app folder to the import path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.services.model_manager import LocalModel, ModelStatus
from app.ui_state import Services, get_services
from app.utils.errors import ZeroShotError

STATUS_TEXT = {
    ModelStatus.NOT_DOWNLOADED: "Not downloaded",
    ModelStatus.DOWNLOADING: "Downloading",
    ModelStatus.READY: "Ready",
    ModelStatus.FAILED: "Failed",
}


def model_row(model: LocalModel) -> dict[str, str]:
    return {
        "Name": model.config.name,
        "Identifier": model.config.id,
        "Status": STATUS_TEXT[model.status],
        "Revision": model.revision or model.config.revision,
        "Path": str(model.path),
        "Details": model.error or "",
    }


def show_model_table(services: Services) -> None:
    st.table([model_row(model) for model in services.models.statuses()])
    st.text(
        f"Model directory: {services.models.model_dir}\nCache location: {services.models.hf_home}"
    )


def download_models(services: Services, model_ids: list[str]) -> None:
    for model_id in model_ids:
        name = services.models.get_config(model_id).name
        try:
            with st.spinner(f"Downloading {name}. Large models can take several minutes."):
                outcome = services.models.download(model_id)
        except ZeroShotError as error:
            st.error(error.user_message)
        else:
            if outcome.already_available:
                st.info(f"{name} is already available, so nothing was downloaded.")
            else:
                st.success(f"{name} was downloaded and is ready.")


st.set_page_config(page_title="Admin")
st.title("Admin")
st.warning("This page has no login. Use it only for local development or trusted environments.")

services = get_services()

st.header("Models")
table_area = st.container()
st.button("Refresh list", help="Reload model status from disk.")

st.subheader("Download")
configs = {config.id: config for config in services.models.configs()}
selected_ids = st.multiselect(
    "Models to download", list(configs), format_func=lambda model_id: configs[model_id].name
)
if st.button("Download selected", disabled=not selected_ids):
    download_models(services, selected_ids)

with table_area:
    show_model_table(services)
