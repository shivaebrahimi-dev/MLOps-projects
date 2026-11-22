"""Hallucination guardrail, answer validation, and prompt-injection screening (User Story 3).

Every rule here is deterministic: it compares the model's draft with the chunks the retriever
actually returned, so it can be unit-tested and gives the same verdict every time (research R6).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from pydantic import ValidationError

from app.models import SAFE_ANSWER, AnswerDraft, RetrievedChunk

# -- prompt injection (FR-014, research R7) ------------------------------------------------------

_INJECTION_PATTERNS = [
    r"\bignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier|preceding)\s+"
    r"(?:instructions?|prompts?|rules|directions)",
    r"\bdisregard\s+(?:all\s+|any\s+|the\s+|your\s+)?(?:previous|prior|above|system|earlier)\b",
    r"\bforget\s+(?:all\s+|your\s+|the\s+)?(?:previous\s+)?instructions\b",
    r"\byou\s+are\s+now\s+(?:a|an|in)\b",
    r"\b(?:reveal|print|show|output|leak|repeat)\s+(?:the\s+|your\s+)?"
    r"(?:system\s+prompt|hidden\s+prompt|api[\s_-]?keys?|secrets?|passwords?|credentials)",
    r"\bnew\s+instructions?\s*:",
    r"\bsystem\s+prompt\b",
    r"\bdo\s+not\s+(?:follow|obey)\s+(?:the|your)\s+(?:rules|instructions)",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE)


def detect_injection(text: str) -> bool:
    """True if the text contains wording aimed at overriding the assistant's instructions."""
    return bool(_INJECTION_RE.search(text))


# -- parsing -------------------------------------------------------------------------------------

INLINE_CITATION_RE = re.compile(r"\[([^\[\]\n]+?#\d+)\]")
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

SAFE_DRAFT = AnswerDraft(status="not_found", answer=SAFE_ANSWER, citations=[], missing_evidence=[])


def safe_draft(missing: Iterable[str] = ()) -> AnswerDraft:
    return AnswerDraft(
        status="not_found", answer=SAFE_ANSWER, citations=[], missing_evidence=list(missing)
    )


def parse_draft(value: Any) -> AnswerDraft | None:
    """Turn a task output (model, dict, or JSON text, possibly fenced) into an AnswerDraft."""
    if isinstance(value, AnswerDraft):
        return value
    if isinstance(value, dict):
        try:
            return AnswerDraft.model_validate(value)
        except ValidationError:
            return None
    if not isinstance(value, str):
        return None
    text = _FENCE_RE.sub("", value.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return AnswerDraft.model_validate(json.loads(text[start : end + 1]))
    except (json.JSONDecodeError, ValidationError):
        return None


# -- validation (FR-011, FR-012) -----------------------------------------------------------------


def inline_citations(answer: str) -> list[str]:
    return [m.strip() for m in INLINE_CITATION_RE.findall(answer)]


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item.strip(), None)
    return [i for i in seen if i]


def check_answer(
    draft: AnswerDraft, retrieved: Iterable[RetrievedChunk]
) -> tuple[AnswerDraft, list[str]]:
    """Validate a draft against the retrieved evidence.

    Returns the normalised draft and a list of problems (empty when the draft passes):

    * citations listed in the answer text are merged into `citations`;
    * every citation must be an evidence chunk id that was shown to the agent;
    * `answered` / `partial` need at least one inline citation; `partial` needs missing evidence;
    * `not_found` is normalised to the exact safe answer with no citations.
    """
    evidence_ids = {c.chunk_id for c in retrieved if c.is_evidence}

    if draft.status == "not_found":
        return safe_draft(draft.missing_evidence), []

    inline = inline_citations(draft.answer)
    citations = _dedupe([*draft.citations, *inline])
    draft = draft.model_copy(update={"citations": citations})
    problems: list[str] = []

    unknown = [c for c in citations if c not in evidence_ids]
    if unknown:
        problems.append(
            "These citations are not ids of the evidence you were given: "
            + ", ".join(unknown)
            + ". Cite only ids that appear in an EVIDENCE block, exactly as written."
        )
    if not citations:
        problems.append(
            "The answer cites nothing. Every claim needs an inline [source#n] citation."
        )
    elif not inline:
        problems.append("Put the citations inline in the answer text, like [source#n].")
    if not draft.answer.strip():
        problems.append("The answer text is empty.")
    if draft.status == "partial" and not draft.missing_evidence:
        problems.append("A partial answer must list what evidence is missing in missing_evidence.")
    return draft, problems


def feedback_message(problems: list[str]) -> str:
    return (
        "Your answer failed the citation check:\n- "
        + "\n- ".join(problems)
        + "\nRewrite the JSON answer so that it passes, using only the evidence. "
        f'If the evidence does not answer the question, reply with status "not_found" and '
        f'answer "{SAFE_ANSWER}".'
    )
