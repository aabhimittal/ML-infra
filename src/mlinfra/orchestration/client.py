"""Clients that the orchestration layer uses to reach the serving layer.

Two interchangeable implementations satisfy the same small protocol:

* :class:`ServingClient` - talks to a running FastAPI server over HTTP (the "connect to an
  external service" story).
* :class:`LocalEngineClient` - calls an in-process engine directly, so demos and tests run
  end-to-end without binding a socket.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from mlinfra.serving.engine import ContinuousBatchingEngine
from mlinfra.serving.schemas import GenerateRequest


class GenerationClient(Protocol):
    async def generate(self, prompt: str, max_tokens: int = 64) -> str: ...


class ServingClient:
    """HTTP client for the ``mlinfra`` serving API."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def generate(self, prompt: str, max_tokens: int = 64) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/generate",
                json={"prompt": prompt, "max_tokens": max_tokens},
            )
            resp.raise_for_status()
            return resp.json()["text"]

    async def health(self) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()


class LocalEngineClient:
    """In-process adapter so the pipeline can drive an engine without HTTP."""

    def __init__(self, engine: ContinuousBatchingEngine) -> None:
        self.engine = engine

    async def generate(self, prompt: str, max_tokens: int = 64) -> str:
        resp = await self.engine.generate(
            GenerateRequest(prompt=prompt, max_tokens=max_tokens)
        )
        return resp.text
