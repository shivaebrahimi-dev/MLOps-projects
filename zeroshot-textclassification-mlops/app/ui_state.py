"""Shared service instances for the Streamlit pages."""

from dataclasses import dataclass

import streamlit as st

from app.services.label_manager import LabelManager
from app.services.model_manager import ModelManager, load_model_configs
from app.utils.config import Settings, load_settings
from app.utils.errors import ZeroShotError
from app.utils.logging_config import configure_logging


@dataclass(frozen=True)
class Services:
    settings: Settings
    models: ModelManager
    labels: LabelManager


@st.cache_resource(show_spinner=False)
def _create_services() -> Services:
    settings = load_settings()
    configure_logging(settings.log_level)
    return Services(
        settings=settings,
        models=ModelManager(
            load_model_configs(settings.models_config), settings.model_dir, settings.hf_home
        ),
        labels=LabelManager(settings.labels_config),
    )


def get_services() -> Services:
    """Return the shared services, or show a clear error and stop the page."""
    try:
        return _create_services()
    except ZeroShotError as error:
        st.error(error.user_message)
        st.stop()
