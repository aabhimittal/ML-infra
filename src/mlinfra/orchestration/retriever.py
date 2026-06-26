"""Thin retriever wrapper over a vector store.

Separating the retriever from the store mirrors LangChain/LlamaIndex: the retriever owns
the query-time policy (top-k, score threshold) while the store owns indexing and similarity.
"""

from __future__ import annotations

from mlinfra.orchestration.loaders import Document
from mlinfra.orchestration.vectorstore import InMemoryVectorStore


class Retriever:
    def __init__(
        self,
        store: InMemoryVectorStore,
        k: int = 3,
        score_threshold: float = 0.0,
    ) -> None:
        self.store = store
        self.k = k
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> list[Document]:
        results = self.store.search(query, k=self.k)
        return [doc for doc, score in results if score >= self.score_threshold]
