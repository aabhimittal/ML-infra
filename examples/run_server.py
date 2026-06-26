"""Launch the mlinfra serving API.

    python examples/run_server.py
    # then, in another shell:
    curl localhost:8000/health
    curl -N -X POST localhost:8000/generate/stream \
         -H 'content-type: application/json' \
         -d '{"prompt": "hello", "max_tokens": 12}'
    curl localhost:8000/metrics
"""

from mlinfra.serving.server import main

if __name__ == "__main__":
    main()
