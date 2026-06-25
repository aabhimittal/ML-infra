"""End-to-end RAG demo running fully in-process (no server, no network).

    python examples/rag_demo.py

Builds a small corpus, indexes it, retrieves for a query, and generates an answer using the
in-process continuous-batching engine.
"""

from __future__ import annotations

import asyncio

from mlinfra.orchestration.client import LocalEngineClient
from mlinfra.orchestration.loaders import InMemoryLoader
from mlinfra.orchestration.pipeline import RAGPipeline
from mlinfra.orchestration.retriever import Retriever
from mlinfra.orchestration.vectorstore import InMemoryVectorStore
from mlinfra.serving.engine import ContinuousBatchingEngine

CORPUS = [
    ("vllm", "vLLM uses PagedAttention to manage the KV cache and enable continuous batching."),
    ("tgi", "Text Generation Inference serves transformer models with token streaming."),
    ("mlflow", "MLflow tracks experiments, parameters, and metrics for ML pipelines."),
    ("zenml", "ZenML structures ML workflows as cached, reproducible pipeline steps."),
]


async def main() -> None:
    store = InMemoryVectorStore()
    store.add(InMemoryLoader(CORPUS).load())
    retriever = Retriever(store, k=2)

    async with ContinuousBatchingEngine() as engine:
        pipeline = RAGPipeline(retriever, LocalEngineClient(engine), max_tokens=16)
        result = await pipeline.run("How does vLLM manage GPU memory?")

    print("Query:   ", result.query)
    print("Contexts:", [c.id for c in result.contexts])
    print("Answer:  ", result.answer)


if __name__ == "__main__":
    asyncio.run(main())
