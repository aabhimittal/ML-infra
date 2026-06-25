"""Inference serving layer (vLLM / HF TGI style).

Public surface:

* :class:`~mlinfra.serving.engine.ContinuousBatchingEngine` - the async scheduler.
* :class:`~mlinfra.serving.backends.ModelBackend` and concrete backends.
* :mod:`~mlinfra.serving.server` - the FastAPI application factory.
"""

from mlinfra.serving.backends import (
    ModelBackend,
    MockModelBackend,
    get_backend,
)
from mlinfra.serving.engine import ContinuousBatchingEngine, EngineConfig
from mlinfra.serving.schemas import (
    GenerateRequest,
    GenerateResponse,
    MetricsSnapshot,
)

__all__ = [
    "ModelBackend",
    "MockModelBackend",
    "get_backend",
    "ContinuousBatchingEngine",
    "EngineConfig",
    "GenerateRequest",
    "GenerateResponse",
    "MetricsSnapshot",
]
