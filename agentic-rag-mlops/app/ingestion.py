"""Upload validation, text extraction, chunking, and indexing (User Story 1).

Uploaded bytes are processed in memory and never written to disk (FR-007, Clarification 3).
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import PurePath

from app.config import SUPPORTED_EXTENSIONS, Settings
from app.errors import (
    EmptyFile,
    FileTooLarge,
    NoExtractableText,
    UnsupportedFileType,
    UploadRejected,
)
from app.knowledge_base import KnowledgeBase
from app.models import Chunk, IngestResult
from app.observability import NoopTracer, Tracer

log = logging.getLogger(__name__)

Page = tuple[int | None, str]  # (1-based page number for PDFs, else None; text)

_SEPARATORS = ("\n\n", "\n", ". ", " ")


# -- validation and extraction -------------------------------------------------------------------


def validate_upload(filename: str, data: bytes, max_bytes: int) -> str:
    """Check type and size. Returns the lower-case extension or raises `UploadRejected`."""
    ext = PurePath(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        shown = ext or "(no extension)"
        raise UnsupportedFileType(
            f"unsupported file type: {shown} (supported: {', '.join(SUPPORTED_EXTENSIONS)})"
        )
    if len(data) == 0:
        raise EmptyFile("empty file (0 bytes)")
    if len(data) > max_bytes:
        raise FileTooLarge(
            f"file too large: {len(data) / 1_048_576:.1f} MB (limit {max_bytes / 1_048_576:.0f} MB)"
        )
    return ext


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def extract_text(filename: str, data: bytes) -> list[Page]:
    """Return the text of a file as pages. Raises `UploadRejected` when there is no usable text."""
    ext = PurePath(filename).suffix.lower()
    if ext == ".pdf":
        pages = _extract_pdf(data)
        if not pages:
            raise NoExtractableText(
                "no extractable text in this PDF (scanned PDFs are not supported)"
            )
        return pages
    text = _decode(data)
    if not text.strip():
        raise EmptyFile("empty file (only whitespace, no text)")
    return [(None, text)]


def _extract_pdf(data: bytes) -> list[Page]:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(i, (page.extract_text() or "")) for i, page in enumerate(reader.pages, start=1)]
    except (PdfReadError, ValueError, KeyError, OSError) as exc:
        raise NoExtractableText(f"no extractable text: the PDF could not be read ({exc})") from exc
    return [(n, text) for n, text in pages if text.strip()]


# -- chunking ------------------------------------------------------------------------------------


def _split(text: str, size: int, separators: Sequence[str]) -> list[str]:
    """Split text into pieces no longer than `size`, preferring the coarsest separator."""
    if len(text) <= size:
        return [text]
    for i, sep in enumerate(separators):
        if sep in text:
            parts = text.split(sep)
            pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
            out: list[str] = []
            for piece in pieces:
                if len(piece) > size:
                    out.extend(_split(piece, size, separators[i + 1 :]))
                elif piece:
                    out.append(piece)
            return out
    return [text[i : i + size] for i in range(0, len(text), size)]


def _tail(text: str, overlap: int) -> str:
    """The last `overlap` characters of `text`, starting at a word boundary when possible."""
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    cut = tail.find(" ")
    return tail[cut + 1 :] if 0 <= cut < len(tail) - 1 else tail


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Recursive character chunking with overlap. Every chunk is at most `chunk_size` long."""
    if not text.strip():
        return []
    overlap = min(chunk_overlap, chunk_size // 2)
    chunks: list[str] = []
    current = ""
    for piece in _split(text, chunk_size, _SEPARATORS):
        if len(current) + len(piece) <= chunk_size:
            current += piece
            continue
        if current.strip():
            chunks.append(current.strip())
        tail = _tail(current, overlap)
        current = tail + piece if len(tail) + len(piece) <= chunk_size else piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def build_chunks(
    source: str,
    pages: Sequence[Page],
    chunk_size: int,
    chunk_overlap: int,
    base_metadata: dict,
) -> list[Chunk]:
    """Chunk each page separately (so PDF chunks keep their page) and number them in order."""
    chunks: list[Chunk] = []
    for page, text in pages:
        for piece in chunk_text(text, chunk_size, chunk_overlap):
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{source}#{index}",
                    source=source,
                    chunk_index=index,
                    text=piece,
                    page=page,
                    metadata=dict(base_metadata),
                )
            )
    return chunks


# -- indexing ------------------------------------------------------------------------------------


def ingest_one(filename: str, data: bytes, kb: KnowledgeBase, settings: Settings) -> IngestResult:
    try:
        ext = validate_upload(filename, data, settings.max_upload_bytes)
        pages = extract_text(filename, data)
    except UploadRejected as exc:
        return IngestResult(filename, "rejected", 0, exc.reason)

    metadata = {
        "file_type": ext.lstrip("."),
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "indexed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    chunks = build_chunks(filename, pages, settings.chunk_size, settings.chunk_overlap, metadata)
    if not chunks:
        return IngestResult(filename, "rejected", 0, "empty file (no text to index)")

    existed = any(s["source"] == filename for s in kb.list_sources())
    stored = kb.upsert_document(filename, chunks)
    message = f"indexed {stored} chunk(s)"
    if existed:
        message += "; replaced the previously indexed version"
    return IngestResult(filename, "indexed", stored, message)


def ingest_files(
    files: Iterable[tuple[str, bytes]],
    kb: KnowledgeBase,
    settings: Settings,
    tracer: Tracer | None = None,
) -> list[IngestResult]:
    """Index several uploads. One result per file, in the order given (FR-004)."""
    tracer = tracer or NoopTracer()
    files = list(files)
    last_index = {name: i for i, (name, _) in enumerate(files)}
    started = time.perf_counter()

    with tracer.observation(
        "ingest",
        as_type="span",
        input={"files": [name for name, _ in files]},
    ) as span:
        results: list[IngestResult] = []
        for i, (name, data) in enumerate(files):
            if last_index[name] != i:
                skipped = "skipped: a later file in this upload has the same name"
                results.append(IngestResult(name, "rejected", 0, skipped))
                continue
            result = ingest_one(name, data, kb, settings)
            duplicates = sum(1 for other, _ in files if other == name) > 1
            if duplicates and result.status == "indexed":
                result = IngestResult(
                    name,
                    result.status,
                    result.chunks,
                    result.message + "; an earlier file with the same name was skipped",
                )
            results.append(result)
            log.info("ingest %s: %s (%s)", name, result.status, result.message)

        span.update(
            output=[
                {"file": r.filename, "status": r.status, "chunks": r.chunks, "message": r.message}
                for r in results
            ],
            metadata={
                "indexed": sum(r.status == "indexed" for r in results),
                "rejected": sum(r.status == "rejected" for r in results),
                "latency_s": round(time.perf_counter() - started, 3),
            },
            level="WARNING" if any(r.status == "rejected" for r in results) else None,
        )
    return results
