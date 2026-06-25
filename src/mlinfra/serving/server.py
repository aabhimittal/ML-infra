"""FastAPI application exposing the continuous-batching engine.

Endpoints mirror a minimal TGI/vLLM-style surface:

* ``GET  /health``           - liveness.
* ``POST /generate``         - blocking completion with timing metadata.
* ``POST /generate/stream``  - server-sent-events token stream.
* ``GET  /metrics``          - live throughput / latency / batch snapshot.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from mlinfra.serving.backends import get_backend
from mlinfra.serving.engine import ContinuousBatchingEngine, EngineConfig
from mlinfra.serving.schemas import GenerateRequest, GenerateResponse, MetricsSnapshot


def create_app(engine: ContinuousBatchingEngine | None = None) -> FastAPI:
    """Application factory so tests can inject a pre-built engine."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await app.state.engine.start()
        try:
            yield
        finally:
            await app.state.engine.stop()

    app = FastAPI(title="mlinfra serving", version="0.1.0", lifespan=lifespan)
    app.state.engine = engine or _engine_from_env()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "backend": app.state.engine.backend.name}

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(request: GenerateRequest) -> GenerateResponse:
        return await app.state.engine.generate(request)

    @app.post("/generate/stream")
    async def generate_stream(request: GenerateRequest) -> StreamingResponse:
        async def event_stream() -> AsyncIterator[bytes]:
            async for token in app.state.engine.stream(request):
                yield f"data: {json.dumps({'token': token})}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/metrics", response_model=MetricsSnapshot)
    async def metrics() -> MetricsSnapshot:
        return app.state.engine.metrics()

    return app


def _engine_from_env() -> ContinuousBatchingEngine:
    backend = get_backend(os.environ.get("MLINFRA_BACKEND", "mock"))
    config = EngineConfig(
        max_batch_size=int(os.environ.get("MLINFRA_MAX_BATCH", "8")),
        batch_tick_s=float(os.environ.get("MLINFRA_TICK_S", "0.005")),
    )
    return ContinuousBatchingEngine(backend=backend, config=config)


app = create_app()


def main() -> None:
    """Console-script entry point: ``mlinfra-server``."""
    import uvicorn

    uvicorn.run(
        "mlinfra.serving.server:app",
        host=os.environ.get("MLINFRA_HOST", "127.0.0.1"),
        port=int(os.environ.get("MLINFRA_PORT", "8000")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
