"""Embedding models behind one small interface, so tests can run without the network."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by does did do for from has have how in is it its of on or that the "
    "this to was were what when where which who why with which can".split()
)


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """OpenAI embeddings (default `text-embedding-3-small`)."""

    def __init__(self, model: str, api_key: str, batch_size: int = 64) -> None:
        from openai import OpenAI

        self.name = f"openai/{model}"
        self._model = model
        self._client = OpenAI(api_key=api_key)
        self._batch_size = batch_size

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = [t.replace("\n", " ") for t in texts[start : start + self._batch_size]]
            response = self._client.embeddings.create(model=self._model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        return vectors


class HashEmbedder:
    """Deterministic bag-of-words embedding for offline tests.

    Each content word is hashed into one of `dim` buckets; vectors are L2-normalised, so cosine
    similarity reflects shared vocabulary. Good enough to check that retrieval finds the document
    that uses the question's words, and fully reproducible.
    """

    def __init__(self, dim: int = 512) -> None:
        self.name = f"hash/{dim}"
        self._dim = dim

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in _TOKEN.findall(text.lower()):
            if token in _STOPWORDS or len(token) < 2:
                continue
            digest = hashlib.md5(token.encode("utf-8")).digest()
            vec[int.from_bytes(digest[:4], "little") % self._dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]
