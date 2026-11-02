"""Administrator page for managing models and candidate labels."""

import sys
from pathlib import Path

# `streamlit run` only adds the app folder to the import path, not the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.services import label_manager
from app.services.model_manager import LocalModel, ModelStatus
from app.ui_state import Services, get_services
from app.utils.errors import ZeroShotError

STATUS_TEXT = {
    ModelStatus.NOT_DOWNLOADED: "Not downloaded",
    ModelStatus.DOWNLOADING: "Downloading",
    ModelStatus.READY: "Ready",
    ModelStatus.FAILED: "Failed",
}
NEW_LABEL_SET = "New label set"


def set_message(area: str, kind: str, text: str) -> None:
    st.session_state[f"{area}_message"] = (kind, text)


def show_message(area: str) -> None:
    kind, text = st.session_state.pop(f"{area}_message", (None, None))
    if kind == "success":
        st.success(text)
    elif kind == "info":
        st.info(text)
    elif kind == "error":
        st.error(text)


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


def delete_model(services: Services, model: LocalModel) -> None:
    st.session_state["confirm_delete"] = False
    try:
        services.classifier.unload(model.config.id)
        services.models.delete(model.config.id)
    except ZeroShotError as error:
        set_message("delete", "error", error.user_message)
    else:
        set_message("delete", "success", f"{model.config.name} was deleted.")


def clear_cache(services: Services) -> None:
    st.session_state["confirm_cache"] = False
    try:
        cleanup = services.models.clear_cache()
    except ZeroShotError as error:
        set_message("cache", "error", error.user_message)
        return
    if cleanup.removed_folders:
        megabytes = cleanup.freed_bytes / 1_000_000
        folders = ", ".join(cleanup.removed_folders)
        set_message("cache", "success", f"Cleared {folders} and freed {megabytes:,.0f} MB.")
    else:
        set_message("cache", "info", "There is no cache to clear.")


def show_delete_section(services: Services) -> None:
    st.subheader("Delete a model")
    deletable = [
        model
        for model in services.models.statuses()
        if model.status in (ModelStatus.READY, ModelStatus.FAILED)
    ]
    if not deletable:
        st.caption("No downloaded models to delete.")
    else:
        selected = st.selectbox(
            "Model to delete", deletable, format_func=lambda model: model.config.name
        )
        confirmed = st.checkbox("I understand this removes the local files.", key="confirm_delete")
        st.button(
            "Delete model",
            disabled=not confirmed,
            on_click=delete_model,
            args=(services, selected),
        )
    show_message("delete")


def show_cache_section(services: Services) -> None:
    st.subheader("Hugging Face cache")
    st.caption("Downloaded models are kept in the model directory, so clearing this keeps them.")
    confirmed = st.checkbox("I understand this clears the download cache.", key="confirm_cache")
    st.button("Clear cache", disabled=not confirmed, on_click=clear_cache, args=(services,))
    show_message("cache")


def add_draft_label() -> None:
    try:
        st.session_state["label_draft"] = list(
            label_manager.add_label(st.session_state["label_draft"], st.session_state["new_label"])
        )
    except ZeroShotError as error:
        set_message("labels", "error", error.user_message)
    st.session_state["new_label"] = ""


def remove_draft_label(label: str) -> None:
    st.session_state["label_draft"] = list(
        label_manager.remove_label(st.session_state["label_draft"], label)
    )


def save_draft(services: Services, choice: str) -> None:
    is_new = choice == NEW_LABEL_SET
    name = st.session_state.get("new_set_name", "") if is_new else choice
    try:
        saved = services.labels.save(name, st.session_state["label_draft"], is_new=is_new)
    except ZeroShotError as error:
        set_message("labels", "error", error.user_message)
        return
    set_message("labels", "success", f"Label set '{saved.name}' was saved.")
    st.session_state["label_set_choice"] = saved.name
    st.session_state["label_draft_for"] = saved.name
    st.session_state["label_draft"] = list(saved.labels)
    st.session_state["new_set_name"] = ""


def show_label_section(services: Services) -> None:
    st.header("Candidate labels")
    try:
        label_sets = {label_set.name: label_set for label_set in services.labels.label_sets()}
    except ZeroShotError as error:
        st.error(error.user_message)
        return

    choice = st.selectbox("Label set", [*label_sets, NEW_LABEL_SET], key="label_set_choice")
    if st.session_state.get("label_draft_for") != choice:
        st.session_state["label_draft_for"] = choice
        st.session_state["label_draft"] = (
            list(label_sets[choice].labels) if choice in label_sets else []
        )

    if choice == NEW_LABEL_SET:
        st.text_input("Name of the new label set", key="new_set_name")

    for index, label in enumerate(st.session_state["label_draft"]):
        text, action = st.columns([4, 1])
        text.text(label)
        action.button(
            "Remove", key=f"remove_label_{index}", on_click=remove_draft_label, args=(label,)
        )

    st.text_input("New label", key="new_label")
    st.button("Add label", on_click=add_draft_label)
    st.button("Save label set", type="primary", on_click=save_draft, args=(services, choice))
    show_message("labels")


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

show_delete_section(services)
show_cache_section(services)

with table_area:
    show_model_table(services)

show_label_section(services)
