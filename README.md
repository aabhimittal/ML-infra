# ML-infra

A compact, **runnable** ML-infrastructure showcase. It implements — in pure, CPU-only
Python — the three layers an MLE platform team owns, modeled on the projects engineers
contribute to daily:

| Layer | Modeled on | What it demonstrates |
|-------|-----------|----------------------|
| **Serving** (`mlinfra.serving`) | vLLM, HF TGI | Async **continuous batching**, KV-cache-style per-request state, token streaming (SSE), live throughput/TTFT metrics |
| **Orchestration** (`mlinfra.orchestration`) | LangChain, LlamaIndex | A **RAG pipeline** from composable parts: loaders/connectors, vector store, retriever, serving client |
| **Tracking** (`mlinfra.tracking`) | MLflow, ZenML | sqlite **experiment tracking** + a **DAG scheduler** with content-hash step caching |
| **CUDA** (`mlinfra.cuda`) | vLLM/TGI custom kernels | Real `.cu` kernels **compiled to PTX/SASS on CPU** via NVRTC; GPU-gated launch path |

The layers compose into one story: the orchestration layer calls the serving layer; the
tracking layer records serving and pipeline metrics.

Everything runs **offline, CPU-only, with no GPU and no API keys**. Heavy integrations
(`transformers`, `sentence-transformers`, `anthropic`, `boto3`) are optional adapters loaded
lazily — they are never required to run the project or its tests.

## Architecture

```mermaid
flowchart LR
    subgraph Orchestration
        L[Loaders / connectors] --> V[Vector store]
        V --> R[Retriever]
        R --> P[RAG pipeline]
    end
    subgraph Serving
        Q[Async request queue] --> B[Continuous-batching engine]
        B --> M[(Metrics: TTFT / tok/s / batch)]
    end
    subgraph Tracking
        T[(Experiment tracker - sqlite)]
        S[DAG scheduler + cache]
    end
    P -->|GenerationClient| B
    B --> T
    S --> R
```

## Quickstart

```bash
make install          # pip install -e ".[dev]"
make test             # pytest across all three layers (offline)
make bench            # end-to-end: scheduler + engine + RAG + tracking report
```

### Run the serving API

```bash
make run-server       # uvicorn on 127.0.0.1:8000
curl localhost:8000/health
curl -N -X POST localhost:8000/generate/stream \
     -H 'content-type: application/json' \
     -d '{"prompt": "hello", "max_tokens": 12}'
curl localhost:8000/metrics
```

### RAG demo (fully in-process)

```bash
python examples/rag_demo.py
```

## Design notes

- **Continuous batching** (`serving/engine.py`): a single async loop admits waiting requests
  into the running batch as they arrive and advances every in-flight request one token per
  tick — so throughput scales with batch size while each request streams independently. This
  is the core idea behind vLLM/TGI, expressed as systems code rather than CUDA kernels.
- **Pluggable backends** (`serving/backends.py`): the default `MockModelBackend` is
  deterministic and dependency-free; `HFModelBackend` and `AnthropicBackend` are optional
  adapters behind `try/except`.
- **Composable RAG** (`orchestration/`): `Loader → VectorStore → Retriever → RAGPipeline`,
  backend-agnostic via a small `GenerationClient` protocol (in-process or HTTP).
- **Caching scheduler** (`tracking/scheduler.py`): `@step` functions wired into a DAG by
  parameter name, run in topological order, with outputs cached by a content hash of the
  step source plus its inputs — re-running unchanged steps is a cache hit.

## CUDA kernels (compile on CPU, launch on GPU)

`mlinfra.cuda` ships real CUDA C++ kernels (`saxpy`, a shared-memory **tiled GEMM**, and a
numerically-stable **fused softmax** — the building blocks of a transformer block). The
compiler toolchain installs as pip wheels, so kernels compile **all the way to PTX and SASS
on a CPU-only machine** — no GPU required:

```bash
pip install -e ".[cuda]"            # NVRTC + ptxas wheels
python examples/cuda_compile_demo.py
# fused_softmax  arch=compute_75  PTX=3467B  ~instrs=102  cubin=5536B (SASS)
# tiled_gemm     arch=compute_75  PTX=4721B  ~instrs=126  cubin=4256B (SASS)
# GPU available for launch: False
```

```python
from mlinfra.cuda import compile_kernel, ptx_to_cubin, gpu_available
res = compile_kernel("tiled_gemm")      # CUDA C++ -> PTX (via NVRTC)
cubin = ptx_to_cubin(res.ptx, "sm_75")  # PTX -> SASS (via ptxas)
```

**What runs where:** compilation (`mlinfra.cuda.compile`) is CPU-only and CI-tested.
*Launching* a kernel (`mlinfra.cuda.runtime`) needs an actual GPU + driver (`libcuda.so`);
those entry points check `gpu_available()` and raise/skip cleanly otherwise, so the code path
is ready for a GPU host without breaking CPU CI.

## Optional extras

```bash
pip install -e ".[hf]"          # transformers backend
pip install -e ".[embeddings]"  # sentence-transformers embeddings
pip install -e ".[anthropic]"   # Anthropic API backend
pip install -e ".[cuda]"        # NVRTC + ptxas (kernel compilation on CPU)
```

## Layout

```
src/mlinfra/serving/        # engine, backends, FastAPI server, schemas
src/mlinfra/orchestration/  # loaders, vector store, retriever, client, pipeline
src/mlinfra/tracking/       # tracker, metrics registry, DAG scheduler
src/mlinfra/cuda/           # .cu kernels, NVRTC/ptxas compiler, GPU-gated runtime
examples/                   # run_server, rag_demo, benchmark, cuda_compile_demo
tests/                      # one suite per layer
```
