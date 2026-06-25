"""An async continuous-batching inference engine.

This is the heart of the serving layer and the part that mirrors how vLLM / HF TGI actually
work: requests are admitted into a running batch *as they arrive* (continuous batching)
rather than waiting for a fixed batch to fill. A single background loop advances every
in-flight request by one decode step per tick, so throughput scales with the batch size
while each request streams tokens independently.

It is intentionally pure-Python and CPU-only: the point is to demonstrate the *systems
design* — scheduling, admission, streaming, KV-cache-style per-request state, and live
metrics — not to run real model kernels. Swap in a real backend (see ``backends.py``) and
the same scheduler applies.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from mlinfra.serving.backends import ModelBackend, MockModelBackend, count_tokens
from mlinfra.serving.schemas import GenerateRequest, GenerateResponse, MetricsSnapshot

_DONE = object()  # sentinel pushed onto a request's output queue when generation completes


@dataclass
class EngineConfig:
    """Tunables for the scheduler."""

    max_batch_size: int = 8
    # Wall-clock cost of one decode step for the whole batch. This is what makes batching
    # visibly pay off: every active request advances one token per tick.
    batch_tick_s: float = 0.005
    max_queue: int = 1024


@dataclass
class _RequestState:
    """Per-request decode state — the engine's analogue of a KV-cache slot.

    In a real engine this would hold the attention key/value tensors for the sequence; here
    it holds the incremental decode cursor, output buffer, and timing so the same lifecycle
    (admit -> decode steps -> evict) is modelled faithfully.
    """

    request: GenerateRequest
    request_id: str
    tokens: list[str] = field(default_factory=list)
    output: asyncio.Queue = field(default_factory=asyncio.Queue)
    admitted_at: float = 0.0
    first_token_at: float | None = None
    finished_at: float | None = None
    _iter: object = None  # the backend's token iterator


@dataclass
class _Metrics:
    requests_total: int = 0
    tokens_generated_total: int = 0
    ttft_sum: float = 0.0
    ttft_count: int = 0
    tps_sum: float = 0.0
    tps_count: int = 0
    batch_size_sum: int = 0
    batch_size_samples: int = 0
    observed_max_batch: int = 0


class ContinuousBatchingEngine:
    """Schedules generation across many concurrent requests with one decode loop."""

    def __init__(
        self,
        backend: ModelBackend | None = None,
        config: EngineConfig | None = None,
    ) -> None:
        self.backend = backend or MockModelBackend()
        self.config = config or EngineConfig()
        self._waiting: asyncio.Queue[_RequestState] = asyncio.Queue(self.config.max_queue)
        self._active: list[_RequestState] = []
        self._metrics = _Metrics()
        self._loop_task: asyncio.Task | None = None
        self._started_at = 0.0
        self._stop = asyncio.Event()

    # -- lifecycle ---------------------------------------------------------------------

    async def start(self) -> None:
        if self._loop_task is not None:
            return
        self._started_at = time.perf_counter()
        self._stop.clear()
        self._loop_task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._loop_task is not None:
            await self._loop_task
            self._loop_task = None

    async def __aenter__(self) -> ContinuousBatchingEngine:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # -- submission --------------------------------------------------------------------

    async def submit(self, request: GenerateRequest) -> _RequestState:
        rid = request.request_id or uuid.uuid4().hex
        state = _RequestState(request=request, request_id=rid)
        await self._waiting.put(state)
        return state

    async def stream(self, request: GenerateRequest) -> AsyncIterator[str]:
        """Submit and yield token fragments as they are produced."""
        state = await self.submit(request)
        while True:
            item = await state.output.get()
            if item is _DONE:
                return
            yield item  # type: ignore[misc]

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Submit and await the full completion, returning timing-rich metadata."""
        state = await self.submit(request)
        while (await state.output.get()) is not _DONE:
            pass
        ttft = (state.first_token_at or state.finished_at or 0.0) - state.admitted_at
        total = (state.finished_at or state.admitted_at) - state.admitted_at
        completion_tokens = len(state.tokens)
        return GenerateResponse(
            request_id=state.request_id,
            text="".join(state.tokens),
            prompt_tokens=count_tokens(request.prompt),
            completion_tokens=completion_tokens,
            time_to_first_token_s=max(ttft, 0.0),
            total_latency_s=max(total, 0.0),
            tokens_per_second=(completion_tokens / total) if total > 0 else 0.0,
        )

    # -- scheduler ---------------------------------------------------------------------

    def _admit(self) -> None:
        """Pull waiting requests into the active batch (continuous batching)."""
        while len(self._active) < self.config.max_batch_size and not self._waiting.empty():
            state = self._waiting.get_nowait()
            state.admitted_at = time.perf_counter()
            state._iter = iter(
                self.backend.generate(
                    state.request.prompt,
                    state.request.max_tokens,
                    state.request.temperature,
                )
            )
            self._active.append(state)
            self._metrics.requests_total += 1

    def _step(self) -> None:
        """Advance every active request by one token; evict finished ones."""
        if self._active:
            n = len(self._active)
            self._metrics.batch_size_sum += n
            self._metrics.batch_size_samples += 1
            self._metrics.observed_max_batch = max(self._metrics.observed_max_batch, n)

        still_active: list[_RequestState] = []
        for state in self._active:
            try:
                token = next(state._iter)  # type: ignore[arg-type]
            except StopIteration:
                self._finish(state)
                continue
            now = time.perf_counter()
            if state.first_token_at is None:
                state.first_token_at = now
                self._metrics.ttft_sum += now - state.admitted_at
                self._metrics.ttft_count += 1
            state.tokens.append(token)
            self._metrics.tokens_generated_total += 1
            state.output.put_nowait(token)
            still_active.append(state)
        self._active = still_active

    def _finish(self, state: _RequestState) -> None:
        state.finished_at = time.perf_counter()
        total = state.finished_at - state.admitted_at
        if total > 0 and state.tokens:
            self._metrics.tps_sum += len(state.tokens) / total
            self._metrics.tps_count += 1
        state.output.put_nowait(_DONE)

    async def _run_loop(self) -> None:
        while not self._stop.is_set() or self._active or not self._waiting.empty():
            self._admit()
            self._step()
            await asyncio.sleep(self.config.batch_tick_s)

    # -- introspection -----------------------------------------------------------------

    def metrics(self) -> MetricsSnapshot:
        m = self._metrics
        return MetricsSnapshot(
            requests_total=m.requests_total,
            requests_in_flight=len(self._active),
            queue_depth=self._waiting.qsize(),
            tokens_generated_total=m.tokens_generated_total,
            avg_time_to_first_token_s=(m.ttft_sum / m.ttft_count) if m.ttft_count else 0.0,
            avg_tokens_per_second=(m.tps_sum / m.tps_count) if m.tps_count else 0.0,
            avg_batch_size=(
                m.batch_size_sum / m.batch_size_samples if m.batch_size_samples else 0.0
            ),
            max_batch_size=m.observed_max_batch,
            uptime_s=time.perf_counter() - self._started_at if self._started_at else 0.0,
        )
