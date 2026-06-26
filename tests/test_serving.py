"""Tests for the serving layer: backends, engine batching/streaming, FastAPI surface."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from mlinfra.serving.backends import MockModelBackend, count_tokens, get_backend
from mlinfra.serving.engine import ContinuousBatchingEngine, EngineConfig
from mlinfra.serving.schemas import GenerateRequest
from mlinfra.serving.server import create_app


def test_mock_backend_is_deterministic():
    backend = MockModelBackend()
    a = list(backend.generate("hello world", max_tokens=8, temperature=0.0))
    b = list(backend.generate("hello world", max_tokens=8, temperature=0.0))
    assert a == b
    assert len(a) == 8
    # Different prompts diverge.
    c = list(backend.generate("a different prompt", max_tokens=8, temperature=0.0))
    assert a != c


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("does-not-exist")


def test_count_tokens():
    assert count_tokens("one two three") == 3
    assert count_tokens("") == 0


async def test_engine_generate_returns_metadata():
    async with ContinuousBatchingEngine() as engine:
        resp = await engine.generate(GenerateRequest(prompt="hi", max_tokens=10))
    assert resp.completion_tokens == 10
    assert resp.prompt_tokens == 1
    assert resp.text
    assert resp.total_latency_s >= 0.0
    assert resp.tokens_per_second > 0.0


async def test_engine_streaming_yields_tokens():
    async with ContinuousBatchingEngine() as engine:
        tokens = [t async for t in engine.stream(GenerateRequest(prompt="hi", max_tokens=5))]
    assert len(tokens) == 5
    assert all(isinstance(t, str) for t in tokens)


async def test_continuous_batching_processes_concurrently():
    # With many concurrent requests and a batch loop, total time should be far less than the
    # serial sum (each tick advances the whole batch by one token).
    config = EngineConfig(max_batch_size=16, batch_tick_s=0.005)
    async with ContinuousBatchingEngine(config=config) as engine:
        reqs = [GenerateRequest(prompt=f"p{i}", max_tokens=20) for i in range(16)]
        await asyncio.gather(*(engine.generate(r) for r in reqs))
        snap = engine.metrics()
    assert snap.requests_total == 16
    assert snap.tokens_generated_total == 16 * 20
    assert snap.max_batch_size > 1  # batching actually happened


def test_fastapi_endpoints():
    engine = ContinuousBatchingEngine(config=EngineConfig(batch_tick_s=0.001))
    app = create_app(engine)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"

        resp = client.post("/generate", json={"prompt": "hello", "max_tokens": 6})
        assert resp.status_code == 200
        body = resp.json()
        assert body["completion_tokens"] == 6

        stream = client.post("/generate/stream", json={"prompt": "hi", "max_tokens": 4})
        assert stream.status_code == 200
        assert stream.text.count("data:") == 5  # 4 tokens + [DONE]

        metrics = client.get("/metrics").json()
        assert metrics["requests_total"] >= 2
