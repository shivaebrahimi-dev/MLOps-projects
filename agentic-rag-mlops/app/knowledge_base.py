"""Persistent vector store (ChromaDB) holding chunks and their metadata.

Chroma stores our own embeddings (`embedding_function=None`), so the embedding model is chosen by
the `Embedder` passed in, and tests can use a deterministic one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.embeddings import Embedder
from app.models import Chunk

log = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(self, path: Path, collection_name: str, embedder: Embedder) -> None:
        self.path = Path(path)
        self.collection_name = collection_name
        self.embedder = embedder
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.path), settings=ChromaSettings(anonymized_telemetry=False)
        )

    # -- collection access ---------------------------------------------------------------------

    def _collection(self):
        # get_or_create on every access: if the folder or collection was removed while the app
        # runs, it is recreated empty instead of crashing (analysis A6).
        try:
            return self._client.get_or_create_collection(
                self.collection_name,
                configuration={"hnsw": {"space": "cosine"}},
                embedding_function=None,
                metadata={"embedder": self.embedder.name},
            )
        except Exception:  # noqa: BLE001 - store was deleted under us; start a fresh client
            log.warning("Vector store at %s was unavailable; recreating it", self.path)
            self.path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.path), settings=ChromaSettings(anonymized_telemetry=False)
            )
            return self._client.get_or_create_collection(
                self.collection_name,
                configuration={"hnsw": {"space": "cosine"}},
                embedding_function=None,
                metadata={"embedder": self.embedder.name},
            )

    # -- writes ----------------------------------------------------------------------------------

    def upsert_document(self, source: str, chunks: Sequence[Chunk]) -> int:
        """Replace every chunk of `source` with `chunks` (FR-005). Returns the number stored."""
        collection = self._collection()
        collection.delete(where={"source": source})
        if not chunks:
            return 0
        embeddings = self.embedder.embed([c.text for c in chunks])
        collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[c.chroma_metadata() for c in chunks],
        )
        return len(chunks)

    def delete_source(self, source: str) -> None:
        self._collection().delete(where={"source": source})

    def clear(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001 - nothing to delete
            pass

    # -- reads -----------------------------------------------------------------------------------

    def count(self) -> int:
        return self._collection().count()

    def list_sources(self) -> list[dict[str, Any]]:
        """One row per indexed document, rebuilt from chunk metadata."""
        data = self._collection().get(include=["metadatas"])
        sources: dict[str, dict[str, Any]] = {}
        for meta in data.get("metadatas") or []:
            name = meta["source"]
            row = sources.setdefault(
                name,
                {
                    "source": name,
                    "file_type": meta.get("file_type"),
                    "chunks": 0,
                    "indexed_at": meta.get("indexed_at"),
                    "size_bytes": meta.get("size_bytes"),
                },
            )
            row["chunks"] += 1
        return sorted(sources.values(), key=lambda r: r["source"].lower())

    def query(self, text: str, k: int) -> list[tuple[Chunk, float]]:
        """Top-k chunks by cosine similarity, as (chunk, score) with score = 1 - distance."""
        collection = self._collection()
        total = collection.count()
        if total == 0:
            return []
        embedding = self.embedder.embed([text])[0]
        result = collection.query(
            query_embeddings=[embedding],
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )
        hits: list[tuple[Chunk, float]] = []
        for chunk_id, doc, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        ):
            extra = {k: v for k, v in meta.items() if k not in {"source", "chunk_index", "page"}}
            chunk = Chunk(
                chunk_id=chunk_id,
                source=meta["source"],
                chunk_index=int(meta["chunk_index"]),
                text=doc,
                page=meta.get("page"),
                metadata=extra,
            )
            hits.append((chunk, 1.0 - float(dist)))
        return hits
