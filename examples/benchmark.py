"""End-to-end benchmark tying all three layers together.

    python examples/benchmark.py

What it shows:

1. A ZenML-style indexing **pipeline** (load -> index -> build retriever) run through the
   scheduler with on-disk caching. Run it twice and the steps become cache hits.
2. Concurrent RAG generation against the continuous-batching **engine**, demonstrating that
   throughput scales with the batch the scheduler assembles.
3. **Experiment tracking**: params + latency/throughput metrics logged to sqlite, plus a
   percentile summary from the metrics registry.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from mlinfra.orchestration.client import LocalEngineClient
from mlinfra.orchestration.loaders import Document, InMemoryLoader
from mlinfra.orchestration.pipeline import RAGPipeline
from mlinfra.orchestration.retriever import Retriever
from mlinfra.orchestration.vectorstore import InMemoryVectorStore
from mlinfra.serving.engine import ContinuousBatchingEngine, EngineConfig
from mlinfra.tracking.metrics import MetricsRegistry
from mlinfra.tracking.scheduler import Pipeline, step
from mlinfra.tracking.tracker import ExperimentTracker

CORPUS = [
    ("vllm", "vLLM uses PagedAttention to manage the KV cache and continuous batching."),
    ("tgi", "Hugging Face TGI serves transformer models with token streaming and sharding."),
    ("mlflow", "MLflow tracks experiments, parameters, and metrics across ML pipelines."),
    ("zenml", "ZenML structures ML workflows as cached, reproducible pipeline steps."),
    ("langchain", "LangChain composes retrievers, prompts, and models into chains."),
    ("llamaindex", "LlamaIndex builds indices over documents for retrieval-augmented queries."),
]

QUERIES = [
    "How does vLLM manage GPU memory?",
    "What does MLflow track?",
    "How are ZenML pipelines cached?",
    "What does a retriever do in LangChain?",
    "How does TGI serve models?",
    "What is an index in LlamaIndex?",
]


# --- indexing pipeline (scheduler + caching) -------------------------------------------

@step
def load(corpus: list[tuple[str, str]]) -> list[Document]:
    return InMemoryLoader(corpus).load()


@step
def index(load: list[Document]) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.add(load)
    return store


@step
def retriever(index: InMemoryVectorStore) -> Retriever:
    return Retriever(index, k=2)


def build_retriever(cache_dir: Path) -> tuple[Retriever, list[str], list[str]]:
    pipe = Pipeline(
        steps=[load, index, retriever],
        inputs={"corpus": CORPUS},
        cache_dir=cache_dir,
    )
    report = pipe.run()
    return report.outputs["retriever"], report.executed, report.cached


# --- generation benchmark --------------------------------------------------------------

async def run_benchmark() -> None:
    cache_dir = Path(".mlinfra_cache")

    print("== Indexing pipeline (run 1) ==")
    retr, executed, cached = build_retriever(cache_dir)
    print(f"  executed={executed} cached={cached}")

    print("== Indexing pipeline (run 2 — expect cache hits) ==")
    _, executed2, cached2 = build_retriever(cache_dir)
    print(f"  executed={executed2} cached={cached2}")

    registry = MetricsRegistry()
    tracker = ExperimentTracker(db_path=str(cache_dir / "mlruns.db"))
    config = EngineConfig(max_batch_size=8, batch_tick_s=0.002)

    async with ContinuousBatchingEngine(config=config) as engine:
        client = LocalEngineClient(engine)
        pipeline = RAGPipeline(retr, client, max_tokens=24)

        with tracker.start_run(experiment="rag-bench", name="mock-backend") as run:
            run.log_params(
                {
                    "backend": engine.backend.name,
                    "max_batch_size": config.max_batch_size,
                    "num_queries": len(QUERIES),
                    "max_tokens": 24,
                }
            )

            wall_start = time.perf_counter()

            async def one(q: str) -> int:
                t0 = time.perf_counter()
                result = await pipeline.run(q)
                registry.observe("latency_s", time.perf_counter() - t0)
                tokens = len(result.answer.split())
                registry.observe("answer_tokens", tokens)
                return tokens

            token_counts = await asyncio.gather(*(one(q) for q in QUERIES))
            wall = time.perf_counter() - wall_start

            total_tokens = sum(token_counts)
            throughput = total_tokens / wall if wall else 0.0
            summary = registry.summary()
            run.log_metrics(
                {
                    "wall_time_s": wall,
                    "throughput_tok_per_s": throughput,
                    "latency_p50_s": summary["latency_s.p50"],
                    "latency_p95_s": summary["latency_s.p95"],
                }
            )
            run_id = run.run_id

        snap = engine.metrics()

    print("\n== Engine metrics ==")
    print(f"  requests_total      : {snap.requests_total}")
    print(f"  tokens_generated    : {snap.tokens_generated_total}")
    print(f"  avg_batch_size      : {snap.avg_batch_size:.2f} (max {snap.max_batch_size})")
    print(f"  avg_ttft_s          : {snap.avg_time_to_first_token_s:.4f}")
    print("\n== Benchmark report ==")
    print(f"  wall_time_s         : {wall:.3f}")
    print(f"  total_answer_tokens : {total_tokens}")
    print(f"  throughput_tok/s    : {throughput:.1f}")
    print(f"  latency p50 / p95 s : {summary['latency_s.p50']:.4f} / {summary['latency_s.p95']:.4f}")

    logged = tracker.get_run(run_id)
    print(f"\n== Tracked run {logged.run_id[:8]} ({logged.experiment}) ==")
    print(f"  params : {logged.params}")
    print(f"  metrics: { {k: round(v, 4) for k, v in logged.metrics.items()} }")


def main() -> None:
    asyncio.run(run_benchmark())


if __name__ == "__main__":
    main()
