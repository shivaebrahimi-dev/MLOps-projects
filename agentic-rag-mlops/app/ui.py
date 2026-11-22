"""Streamlit UI: Knowledge (upload and index) and Ask (cited answers) tabs.

Run with:  streamlit run app/ui.py
This is the only module that imports Streamlit (Constitution III); see specs/.../contracts/ui.md.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from app.config import APP_VERSION, MAX_QUESTION_CHARS, get_settings  # noqa: E402
from app.crew import answer_question  # noqa: E402
from app.errors import InvalidQuestion  # noqa: E402
from app.ingestion import ingest_files  # noqa: E402
from app.models import AnswerResult  # noqa: E402
from app.services import Services, build_services  # noqa: E402

STATUS_BADGE = {
    "answered": ("Answered", "green", ":material/check_circle:"),
    "partial": ("Partial", "orange", ":material/rule:"),
    "not_found": ("Not found in knowledge", "gray", ":material/search_off:"),
    "error": ("Error", "red", ":material/error:"),
}


@st.cache_resource(show_spinner="Starting services…")
def get_services() -> Services:
    return build_services(get_settings())


def session_id() -> str:
    """Random, non-identifying id for this browser session (FR-015)."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


# -- Knowledge tab -------------------------------------------------------------------------------


def knowledge_tab(svc: Services) -> None:
    st.subheader("Upload and index knowledge files")
    st.caption(
        "Supported: .txt, .md, .pdf (up to "
        f"{svc.settings.max_upload_mb:g} MB each). Files are processed in memory, not saved; "
        "only their text chunks and metadata are stored."
    )
    uploads = st.file_uploader(
        "Choose one or more files",
        accept_multiple_files=True,
        key=f"uploader-{st.session_state.get('uploader_round', 0)}",
    )
    can_index = bool(uploads) and svc.llm_factory is not None
    if st.button("Index files", type="primary", disabled=not can_index, icon=":material/upload:"):
        with st.spinner("Parsing, chunking and embedding…"):
            results = ingest_files(
                [(f.name, f.getvalue()) for f in uploads], svc.kb, svc.settings, svc.tracer
            )
            svc.tracer.flush()
        st.session_state.last_ingest = results
        st.session_state.uploader_round = st.session_state.get("uploader_round", 0) + 1
        st.rerun()

    results = st.session_state.get("last_ingest")
    if results:
        indexed = [r for r in results if r.status == "indexed"]
        rejected = [r for r in results if r.status == "rejected"]
        if indexed:
            st.success(
                f"Indexed {len(indexed)} file(s), {sum(r.chunks for r in indexed)} chunk(s).",
                icon=":material/check_circle:",
            )
        if rejected:
            st.error(
                f"{len(rejected)} file(s) rejected. See the reasons below.", icon=":material/block:"
            )
        st.dataframe(
            [
                {
                    "File": r.filename,
                    "Status": "✅ Indexed" if r.status == "indexed" else "❌ Rejected",
                    "Chunks": r.chunks,
                    "Message": r.message,
                }
                for r in results
            ],
            hide_index=True,
            use_container_width=True,
        )

    st.divider()
    sources = svc.kb.list_sources()
    st.subheader(f"Knowledge base ({len(sources)} file(s), {svc.kb.count()} chunk(s))")
    if not sources:
        st.info("The knowledge base is empty.", icon=":material/inventory_2:")
        return
    st.dataframe(
        [
            {
                "File": s["source"],
                "Type": s["file_type"],
                "Chunks": s["chunks"],
                "Indexed at (UTC)": s["indexed_at"],
            }
            for s in sources
        ],
        hide_index=True,
        use_container_width=True,
    )
    confirm = st.checkbox("I want to delete every indexed chunk")
    if st.button("Clear knowledge base", disabled=not confirm, icon=":material/delete:"):
        svc.kb.clear()
        st.session_state.pop("last_ingest", None)
        st.session_state.pop("last_answer", None)
        st.rerun()


# -- Ask tab -------------------------------------------------------------------------------------


def ask_tab(svc: Services) -> None:
    st.subheader("Ask a question")
    empty = svc.kb.count() == 0
    if empty:
        st.info("Upload files in the Knowledge tab first.", icon=":material/arrow_back:")
    with st.form("ask", clear_on_submit=False, border=False):
        question = st.text_area(
            "Your question",
            max_chars=MAX_QUESTION_CHARS,
            disabled=empty,
            placeholder="e.g. Which MinIO bucket stores the Fashion-MNIST models?",
        )
        submitted = st.form_submit_button(
            "Ask", type="primary", disabled=empty or svc.llm_factory is None, icon=":material/send:"
        )

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.status("Agents are working…", expanded=True) as status:
                st.write("1. Retriever agent: searching the knowledge base")
                st.write("2. Answer/Verifier agent: drafting and checking citations")
                try:
                    result = answer_question(
                        question,
                        kb=svc.kb,
                        settings=svc.settings,
                        llm_factory=svc.llm_factory,
                        tracer=svc.tracer,
                        session_id=session_id(),
                    )
                except InvalidQuestion as exc:
                    status.update(label=str(exc), state="error")
                    return
                status.update(
                    label=f"Finished in {result.latency_s:.1f} s",
                    state="error" if result.status == "error" else "complete",
                    expanded=False,
                )
            st.session_state.last_answer = result

    result = st.session_state.get("last_answer")
    if result is not None:
        render_answer(result)


def render_answer(result: AnswerResult) -> None:
    label, color, icon = STATUS_BADGE[result.status]
    st.markdown(f"**Question:** {result.question}")
    st.badge(label, color=color, icon=icon)
    for notice in result.notices:
        st.caption(notice)

    if result.status == "error":
        st.error(f"The question could not be answered. {result.error}")
    else:
        st.markdown(result.answer)

    if result.missing_evidence and result.status != "error":
        st.markdown("**Missing evidence**")
        for item in result.missing_evidence:
            st.markdown(f"- {item}")

    if result.citations:
        st.markdown(f"**Sources ({len(result.citations)})**")
        for c in result.citations:
            page = f", page {c.chunk.page}" if c.chunk.page else ""
            with st.expander(f"[{c.chunk_id}]  {c.source}{page} · score {c.score:.2f}"):
                st.text(c.chunk.text)

    with st.expander("Run details"):
        usage = result.token_usage or {}
        cols = st.columns(4)
        cols[0].metric("Latency", f"{result.latency_s:.1f} s")
        cols[1].metric("Tokens", usage.get("total", "n/a"))
        cols[2].metric("Guardrail attempts", result.guardrail.attempts)
        cols[3].metric("Chunks retrieved", len(result.retrieved))
        st.write(f"Model: `{result.model}` · Request id: `{result.request_id}`")
        if result.guardrail.fallback:
            st.warning(
                "The draft failed the citation check after all retries; the safe answer was used."
            )
        if result.injection_flags:
            st.warning(
                "Possible prompt injection in retrieved text (treated as data, not followed): "
                + ", ".join(result.injection_flags)
            )
        if result.trace_id:
            link = f" · [open in Langfuse]({result.trace_url})" if result.trace_url else ""
            st.write(f"Langfuse trace id: `{result.trace_id}`{link}")
        else:
            st.write("Tracing is off, so this run has no trace.")


# -- page ----------------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Agentic RAG Assistant", page_icon=":material/menu_book:")
    svc = get_services()
    st.title("Agentic RAG Knowledge Assistant")
    st.caption(
        f"v{APP_VERSION} · CrewAI · model `{svc.settings.openai_model}` · {svc.tracer.description}"
    )
    for problem in svc.problems:
        st.error(problem, icon=":material/key_off:")

    knowledge, ask = st.tabs(["Knowledge", "Ask"], key="main-tabs")
    with knowledge:
        knowledge_tab(svc)
    with ask:
        ask_tab(svc)


main()
