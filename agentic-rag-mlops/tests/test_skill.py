"""The student skill has the required frontmatter, and both copies are identical (FR-023)."""

from __future__ import annotations

import yaml

from app.config import PROJECT_ROOT

PRIMARY = PROJECT_ROOT / ".agents" / "skills" / "rag-evaluation-report" / "SKILL.md"
MIRROR = PROJECT_ROOT / ".claude" / "skills" / "rag-evaluation-report" / "SKILL.md"


def frontmatter(path):
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    block = text.split("---\n", 2)[1]
    return yaml.safe_load(block), text


def test_skill_frontmatter_has_the_required_values():
    meta, body = frontmatter(PRIMARY)
    assert meta["name"] == "rag-evaluation-report"
    assert meta["user-invocable"] is True
    assert meta["disable-model-invocation"] is True
    assert "explicitly" in meta["description"]
    assert "Only when the user explicitly asks" in body
    assert "Never invent a value" in body and "Unavailable" in body


def test_skill_is_read_only_except_for_writing_the_report():
    meta, _ = frontmatter(PRIMARY)
    tools = {t.strip() for t in meta["allowed-tools"].split(",")}
    assert tools == {"Read", "Glob", "Grep", "Write"}


def test_claude_code_copy_is_identical():
    assert MIRROR.read_bytes() == PRIMARY.read_bytes()
