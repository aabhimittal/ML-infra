"""Pluggable model backends.

The default :class:`MockModelBackend` is fully deterministic and dependency-free so the
whole system runs anywhere (no GPU, no network, no API key). Real integrations
(``HFModelBackend``, ``AnthropicBackend``) are optional adapters imported lazily so that
missing heavy dependencies never break ``import mlinfra``.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator

_WORD_RE = re.compile(r"\S+")

# A small, fixed vocabulary keeps mock output readable and deterministic.
_VOCAB = (
    "the model serves tokens through a batched scheduler while tracking latency and "
    "throughput across requests so operators can reason about tail behavior and "
    "capacity under load in production".split()
)


def count_tokens(text: str) -> int:
    """Whitespace token count — a cheap, deterministic stand-in for a real tokenizer."""
    return len(_WORD_RE.findall(text))


class ModelBackend(ABC):
    """Abstract token-streaming backend.

    Implementations yield one token (string fragment, including any leading space) at a
    time. The engine is responsible for batching, scheduling, and metrics — a backend only
    has to know how to decode.
    """

    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Iterator[str]:
        """Yield up to ``max_tokens`` token fragments for ``prompt``."""
        raise NotImplementedError


class MockModelBackend(ModelBackend):
    """Deterministic, CPU-only backend.

    Output is seeded from a hash of the prompt so results are reproducible in tests while
    still varying by input. This exists to exercise the *infrastructure* (batching,
    streaming, metrics) without pulling in model weights.
    """

    name = "mock"

    def __init__(self, vocab: tuple[str, ...] | list[str] | None = None) -> None:
        self._vocab = tuple(vocab) if vocab else tuple(_VOCAB)

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> Iterator[str]:
        seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
        for i in range(max_tokens):
            word = self._vocab[(seed + i * 2654435761) % len(self._vocab)]
            yield (" " if i else "") + word


def _load_hf_backend(model: str) -> ModelBackend:
    """Optional Hugging Face ``transformers`` backend (requires the ``hf`` extra)."""
    try:
        from transformers import pipeline  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when extra is absent
        raise RuntimeError(
            "HFModelBackend requires the 'hf' extra: pip install mlinfra[hf]"
        ) from exc

    generator = pipeline("text-generation", model=model)

    class HFModelBackend(ModelBackend):
        name = f"hf:{model}"

        def generate(self, prompt: str, max_tokens: int, temperature: float) -> Iterator[str]:
            out = generator(
                prompt,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
            )
            text = out[0]["generated_text"][len(prompt) :]
            for i, word in enumerate(_WORD_RE.findall(text)):
                yield (" " if i else "") + word

    return HFModelBackend()


def _load_anthropic_backend(model: str) -> ModelBackend:
    """Optional Anthropic API backend (requires the ``anthropic`` extra + an API key)."""
    try:
        import anthropic  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when extra is absent
        raise RuntimeError(
            "AnthropicBackend requires the 'anthropic' extra: pip install mlinfra[anthropic]"
        ) from exc

    client = anthropic.Anthropic()

    class AnthropicBackend(ModelBackend):
        name = f"anthropic:{model}"

        def generate(self, prompt: str, max_tokens: int, temperature: float) -> Iterator[str]:
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for chunk in stream.text_stream:
                    yield chunk

    return AnthropicBackend()


def get_backend(kind: str = "mock", **kwargs: object) -> ModelBackend:
    """Factory mapping a backend name to an instance.

    ``kind="mock"`` (default) is always available. ``"hf"`` and ``"anthropic"`` require
    their respective optional extras and are loaded lazily so importing this module never
    fails on a machine without them.
    """
    if kind == "mock":
        return MockModelBackend()
    if kind == "hf":
        return _load_hf_backend(str(kwargs.get("model", "sshleifer/tiny-gpt2")))
    if kind == "anthropic":
        return _load_anthropic_backend(str(kwargs.get("model", "claude-opus-4-8")))
    raise ValueError(f"Unknown backend kind: {kind!r}")
