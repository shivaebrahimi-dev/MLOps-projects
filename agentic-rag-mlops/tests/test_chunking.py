"""Chunking and metadata preservation (FR-003, FR-005)."""

from __future__ import annotations

from app.ingestion import build_chunks, chunk_text, ingest_files
from tests.pdf_helpers import text_pdf

LONG_TEXT = "\n\n".join(
    f"Paragraph {i}. " + " ".join(f"word{i}_{j}" for j in range(40)) for i in range(12)
)


def test_chunks_respect_size_and_cover_the_text() -> None:
    chunks = chunk_text(LONG_TEXT, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(0 < len(c) <= 300 for c in chunks)
    # Nothing is lost: every word of the source appears in some chunk.
    joined = " ".join(chunks)
    assert all(w in joined for w in LONG_TEXT.split())


def test_consecutive_chunks_overlap() -> None:
    chunks = chunk_text(LONG_TEXT, chunk_size=300, chunk_overlap=60)
    overlapping = sum(1 for a, b in zip(chunks, chunks[1:], strict=False) if a.split()[-1] in b)
    assert overlapping >= len(chunks) - 2


def test_zero_overlap_and_short_text() -> None:
    assert chunk_text("short text", 300, 0) == ["short text"]
    assert chunk_text("   ", 300, 0) == []


def test_a_single_long_word_is_hard_split() -> None:
    chunks = chunk_text("x" * 1000, chunk_size=300, chunk_overlap=0)
    assert all(len(c) <= 300 for c in chunks) and "".join(chunks) == "x" * 1000


def test_chunk_ids_and_metadata_are_preserved() -> None:
    chunks = build_chunks(
        "My Notes.md", [(None, LONG_TEXT)], 300, 50, {"file_type": "md", "sha256": "abc"}
    )
    assert [c.chunk_id for c in chunks] == [f"My Notes.md#{i}" for i in range(len(chunks))]
    assert all(c.source == "My Notes.md" and c.page is None for c in chunks)
    assert all(c.metadata["sha256"] == "abc" for c in chunks)
    meta = chunks[0].chroma_metadata()
    assert meta["source"] == "My Notes.md" and meta["chunk_index"] == 0 and "page" not in meta


def test_pdf_chunks_keep_page_numbers_and_never_cross_pages() -> None:
    chunks = build_chunks("doc.pdf", [(1, "alpha " * 30), (2, "beta " * 30)], 100, 20, {})
    pages = {c.page for c in chunks}
    assert pages == {1, 2}
    assert all(("alpha" in c.text) != ("beta" in c.text) for c in chunks)


def test_metadata_survives_the_round_trip_through_the_store(kb, settings) -> None:
    pdf = text_pdf(["The LeNet-5 network is trained on Fashion-MNIST.", "MLflow tracks runs."])
    [result] = ingest_files([("Course Notes é.pdf", pdf)], kb, settings)
    assert result.status == "indexed"
    hits = kb.query("MLflow tracks runs", k=5)
    chunk = next(c for c, _ in hits if "MLflow" in c.text)
    assert chunk.source == "Course Notes é.pdf"
    assert chunk.page == 2
    assert chunk.chunk_id.startswith("Course Notes é.pdf#")
    assert chunk.metadata["file_type"] == "pdf" and len(chunk.metadata["sha256"]) == 64


def test_reupload_replaces_instead_of_duplicating(kb, settings) -> None:
    ingest_files([("a.txt", LONG_TEXT.encode())], kb, settings)
    first = kb.count()
    [again] = ingest_files([("a.txt", LONG_TEXT.encode())], kb, settings)
    assert kb.count() == first
    assert "replaced" in again.message
    [shorter] = ingest_files([("a.txt", b"Only one short paragraph now.")], kb, settings)
    assert shorter.chunks == 1 and kb.count() == 1


def test_same_name_twice_in_one_upload_keeps_the_later_file(kb, settings) -> None:
    results = ingest_files(
        [("dup.txt", b"first version about MinIO"), ("dup.txt", b"second version about MLflow")],
        kb,
        settings,
    )
    assert results[0].status == "rejected" and "later file" in results[0].message
    assert results[1].status == "indexed" and "same name" in results[1].message
    [(chunk, _)] = kb.query("version", k=1)
    assert "second version" in chunk.text
