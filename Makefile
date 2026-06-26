.PHONY: install test lint run-server bench clean

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

lint:
	python -m ruff check src tests examples

run-server:
	python examples/run_server.py

bench:
	python examples/benchmark.py

cuda-demo:
	python examples/cuda_compile_demo.py

numba-demo:
	python examples/numba_compile_demo.py

kernel-bench:
	python examples/kernel_bench.py

clean:
	rm -rf .mlinfra_cache mlruns.db .pytest_cache **/__pycache__ build *.egg-info
