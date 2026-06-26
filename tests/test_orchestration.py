"""Tests for the orchestration layer: loaders, vector store, retriever, RAG pipeline."""

from __future__ import annotations

from pathlib import Path

from mlinfra.orchestration.client import LocalEngineClient
from mlinfra.orchestration.loaders import DirectoryLoader, InMemoryLoader
from mlinfra.orchestration.pipeline import RAGPipeline
from mlinfra.orchestration.retriever import Retriever
from mlinfra.orchestration.vectorstore import InMemoryVectorStore, cosine, hash_embedding
from mlinfra.serving.engine import ContinuousBatchingEngine

DOCS = [
    ("gpu", "GPU memory and the KV cache determine inference batch size."),
    ("cooking", "A good risotto needs slow stirring and warm stock."),
    ("tracking", "Experiment tracking records params and metrics for each run."),
]


def test_hash_embedding_is_deterministic_and_normalized():
    a = hash_embedding("hello world")
    b = hash_embedding("hello world")
    assert a == b
    assert abs(cosine(a, a) - 1.0) < 1e-9


def test_vector_store_retrieves_relevant_doc():
    store = InMemoryVectorStore()
    store.add(InMemoryLoader(DOCS).load())
    assert len(store) == 3
    results = store.search("how big can the inference batch be on the GPU?", k=1)
    assert results[0][0].id == "gpu"


def test_retriever_threshold_filters():
    store = InMemoryVectorStore()
    store.add(InMemoryLoader(DOCS).load())
    retriever = Retriever(store, k=3, score_threshold=2.0)  # impossibly high
    assert retriever.retrieve("anything") == []


def test_directory_loader(tmp_path: Path):
    (tmp_path / "a.txt").write_text("first document", encoding="utf-8")
    (tmp_path / "b.txt").write_text("second document", encoding="utf-8")
    docs = DirectoryLoader(tmp_path).load()
    assert {d.id for d in docs} == {"a.txt", "b.txt"}


async def test_rag_pipeline_end_to_end():
    store = InMemoryVectorStore()
    store.add(InMemoryLoader(DOCS).load())
    retriever = Retriever(store, k=2)

    async with ContinuousBatchingEngine() as engine:
        pipeline = RAGPipeline(retriever, LocalEngineClient(engine), max_tokens=12)
        result = await pipeline.run("Tell me about GPU memory and batching")

    assert result.contexts  # retrieved something
    assert result.contexts[0].id == "gpu"
    assert "Context:" in result.prompt
    assert result.answer  # engine produced an answer
