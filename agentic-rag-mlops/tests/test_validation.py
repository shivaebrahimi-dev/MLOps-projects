"""File type and empty-file validation (FR-001, FR-002, SC-002)."""

from __future__ import annotations

import pytest

from app.errors import EmptyFile, FileTooLarge, NoExtractableText, UnsupportedFileType
from app.ingestion import extract_text, ingest_files, validate_upload
from tests.pdf_helpers import blank_pdf, text_pdf

MAX = 1024 * 1024


@pytest.mark.parametrize("name", ["notes.docx", "image.png", "data.csv", "README", "archive.zip"])
def test_unsupported_types_are_rejected(name: str) -> None:
    with pytest.raises(UnsupportedFileType, match="unsupported file type"):
        validate_upload(name, b"some content", MAX)


@pytest.mark.parametrize("name", ["a.txt", "b.MD", "c.Pdf"])
def test_supported_types_pass_and_extension_is_case_insensitive(name: str) -> None:
    assert validate_upload(name, b"x", MAX) in {".txt", ".md", ".pdf"}


def test_empty_file_is_rejected() -> None:
    with pytest.raises(EmptyFile, match="empty file"):
        validate_upload("empty.txt", b"", MAX)


@pytest.mark.parametrize("content", [b"   ", b"\n\n\t  \r\n", "  ".encode()])
def test_whitespace_only_file_is_rejected(content: bytes) -> None:
    with pytest.raises(EmptyFile, match="empty file"):
        extract_text("blank.txt", content)


def test_oversized_file_is_rejected() -> None:
    with pytest.raises(FileTooLarge, match="too large"):
        validate_upload("big.txt", b"x" * (MAX + 1), MAX)


def test_pdf_without_text_is_rejected() -> None:
    with pytest.raises(NoExtractableText, match="no extractable text"):
        extract_text("scan.pdf", blank_pdf())


def test_corrupt_pdf_is_rejected_with_reason() -> None:
    with pytest.raises(NoExtractableText):
        extract_text("broken.pdf", b"%PDF-1.4 this is not really a pdf")


def test_text_and_markdown_are_decoded() -> None:
    assert extract_text("a.txt", "Café latte".encode()) == [(None, "Café latte")]
    assert extract_text("a.md", b"# Title\n\nBody") == [(None, "# Title\n\nBody")]
    # Latin-1 fallback
    assert extract_text("a.txt", "naïve".encode("latin-1"))[0][1] == "naïve"


def test_pdf_text_is_extracted_per_page() -> None:
    pages = extract_text("doc.pdf", text_pdf(["First page text", "Second page text"]))
    assert [p for p, _ in pages] == [1, 2]
    assert "Second page" in pages[1][1]


def test_ingest_reports_each_file_and_stores_only_valid_ones(kb, settings) -> None:
    results = ingest_files(
        [
            ("good.txt", b"MinIO stores the trained model artifacts."),
            ("empty.txt", b""),
            ("notes.docx", b"PK\x03\x04"),
            ("guide.md", b"# Guide\n\nStreamlit shows the classification page."),
        ],
        kb,
        settings,
    )
    by_name = {r.filename: r for r in results}
    assert by_name["good.txt"].status == "indexed" and by_name["good.txt"].chunks >= 1
    assert by_name["guide.md"].status == "indexed"
    assert by_name["empty.txt"].status == "rejected"
    assert "empty file" in by_name["empty.txt"].message
    assert by_name["notes.docx"].status == "rejected"
    assert "unsupported file type: .docx" in by_name["notes.docx"].message
    assert {s["source"] for s in kb.list_sources()} == {"good.txt", "guide.md"}
