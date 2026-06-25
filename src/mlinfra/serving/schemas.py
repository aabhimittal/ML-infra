"""Pydantic request/response and metrics schemas for the serving layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """A single text-generation request."""

    prompt: str = Field(..., description="Input prompt to condition generation on.")
    max_tokens: int = Field(32, ge=1, le=4096, description="Maximum number of tokens to emit.")
    temperature: float = Field(
        0.0, ge=0.0, le=2.0, description="Sampling temperature (mock backend is deterministic)."
    )
    request_id: str | None = Field(
        None, description="Optional client-supplied id; one is generated if omitted."
    )


class GenerateResponse(BaseModel):
    """The completed generation plus per-request timing."""

    request_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    time_to_first_token_s: float = Field(
        ..., description="Latency from admission to first emitted token."
    )
    total_latency_s: float
    tokens_per_second: float


class MetricsSnapshot(BaseModel):
    """A point-in-time view of engine health, mirroring vLLM/TGI dashboards."""

    requests_total: int
    requests_in_flight: int
    queue_depth: int
    tokens_generated_total: int
    avg_time_to_first_token_s: float
    avg_tokens_per_second: float
    avg_batch_size: float
    max_batch_size: int
    uptime_s: float
