import ast

import pytest

from app.utils.config import PROJECT_ROOT

NON_UI_FILES = sorted(
    path
    for folder in ("app/services", "app/utils")
    for path in (PROJECT_ROOT / folder).glob("*.py")
)


def imported_modules(path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def test_non_ui_files_were_found():
    assert any(path.name == "model_manager.py" for path in NON_UI_FILES)


@pytest.mark.parametrize("path", NON_UI_FILES, ids=lambda path: path.name)
def test_services_and_utils_do_not_import_streamlit(path):
    roots = {module.split(".")[0] for module in imported_modules(path)}

    assert "streamlit" not in roots
