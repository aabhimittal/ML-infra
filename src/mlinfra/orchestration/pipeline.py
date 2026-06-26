"""The RAG pipeline: load -> index -> retrieve -> prompt -> generate.

This wires the orchestration components together and is the layer a product team would
actually call. It is deliberately backend-agnostic: it accepts any
:class:`~mlinfra.orchestration.client.GenerationClient`, so the same pipeline runs against an
in-process engine (tests/benchmark) or a remote HTTP server (production).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mlinfra.orchestration.client import GenerationClient
from mlinfra.orchestration.loaders import Document
from mlinfra.orchestration.retriever import Retriever


@dataclass
class RAGResult:
    query: str
    answer: str
    contexts: list[Document] = field(default_factory=list)
    prompt: str = ""


_PROMPT_TEMPLATE = (
    "Use the following context to answer the question.\n\n"
    "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
)


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        client: GenerationClient,
        max_tokens: int = 64,
        prompt_template: str = _PROMPT_TEMPLATE,
    ) -> None:
        self.retriever = retriever
        self.client = client
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template

    def build_prompt(self, query: str, contexts: list[Document]) -> str:
        joined = "\n\n".join(f"[{i + 1}] {d.text}" for i, d in enumerate(contexts))
        return self.prompt_template.format(context=joined or "(none)", query=query)

    async def run(self, query: str) -> RAGResult:
        contexts = self.retriever.retrieve(query)
        prompt = self.build_prompt(query, contexts)
        answer = await self.client.generate(prompt, max_tokens=self.max_tokens)
        return RAGResult(query=query, answer=answer, contexts=contexts, prompt=prompt)
