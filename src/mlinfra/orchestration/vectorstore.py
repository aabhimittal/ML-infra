"""A tiny in-memory vector store with deterministic, offline embeddings.

The default embedding is a hashed bag-of-words projected into a fixed-dimension vector. It
is deliberately dependency-free and reproducible so retrieval behaves identically in CI and
on a laptop. For real semantic search, pass any callable ``embed(text) -> list[float]`` (for
example a ``sentence-transformers`` model via the ``embeddings`` extra).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

from mlinfra.orchestration.loaders import Document

_TOKEN_RE = re.compile(r"[a-z0-9]+")
EmbeddingFn = Callable[[str], list[float]]


def hash_embedding(text: str, dim: int = 256) -> list[float]:
    """Deterministic bag-of-words embedding via the hashing trick, L2-normalized."""
    vec = [0.0] * dim
    for tok in _TOKEN_RE.findall(text.lower()):
        idx = hash_token(tok) % dim
        sign = 1.0 if (hash_token(tok) >> 16) & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def hash_token(tok: str) -> int:
    """Stable, process-independent hash (Python's ``hash`` is salted per run)."""
    h = 2166136261
    for ch in tok:
        h = (h ^ ord(ch)) * 16777619 & 0xFFFFFFFF
    return h


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # inputs are L2-normalized


@dataclass
class _Entry:
    document: Document
    embedding: list[float]


class InMemoryVectorStore:
    """Brute-force cosine-similarity store — clear and exact for modest corpora."""

    def __init__(self, embed: EmbeddingFn | None = None, dim: int = 256) -> None:
        self.dim = dim
        # Stored as an optional callable (not a lambda) so the store stays picklable for the
        # scheduler's on-disk cache when the default hashing embedder is used.
        self._embed_fn = embed
        self._entries: list[_Entry] = []

    def embed(self, text: str) -> list[float]:
        if self._embed_fn is not None:
            return self._embed_fn(text)
        return hash_embedding(text, self.dim)

    def add(self, documents: list[Document]) -> None:
        for doc in documents:
            self._entries.append(_Entry(document=doc, embedding=self.embed(doc.text)))

    def search(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        q = self.embed(query)
        scored = [(e.document, cosine(q, e.embedding)) for e in self._entries]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]

    def __len__(self) -> int:
        return len(self._entries)
