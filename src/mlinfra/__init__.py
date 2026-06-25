"""mlinfra: a compact, runnable ML-infrastructure showcase.

Three layers that compose into one story:

* ``mlinfra.serving``       - an async continuous-batching inference engine + FastAPI server
                              (vLLM / HF TGI style).
* ``mlinfra.orchestration`` - a RAG pipeline built from composable components that calls the
                              serving layer (LangChain / LlamaIndex style).
* ``mlinfra.tracking``      - experiment tracking + a DAG step scheduler with caching
                              (MLflow / ZenML style).

Everything runs offline and CPU-only by default; GPU/network integrations are optional.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
