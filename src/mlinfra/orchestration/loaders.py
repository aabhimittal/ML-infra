"""Document loaders / connectors.

These mirror the LangChain/LlamaIndex "loader" pattern: a small interface that pulls raw
content from some source into a uniform :class:`Document`. A local-directory loader is the
offline default; an S3-style stub shows where a real AWS connector would plug in.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Document:
    """A unit of retrievable content with arbitrary metadata."""

    id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


class InMemoryLoader:
    """Load documents from in-process strings — handy for tests and demos."""

    def __init__(self, docs: Iterable[tuple[str, str]]) -> None:
        self._docs = [Document(id=did, text=text) for did, text in docs]

    def load(self) -> list[Document]:
        return list(self._docs)


class DirectoryLoader:
    """Load every text file under a directory as a :class:`Document`."""

    def __init__(self, path: str | Path, glob: str = "**/*.txt") -> None:
        self.path = Path(path)
        self.glob = glob

    def load(self) -> list[Document]:
        docs: list[Document] = []
        for file in sorted(self.path.glob(self.glob)):
            if file.is_file():
                docs.append(
                    Document(
                        id=str(file.relative_to(self.path)),
                        text=file.read_text(encoding="utf-8"),
                        metadata={"source": str(file)},
                    )
                )
        return docs


class S3Loader:
    """Stub for an AWS S3 connector (requires ``boto3``; not needed for the demo).

    Kept as a thin, documented seam so the orchestration layer's "connect to external
    storage" story is concrete without making ``boto3`` a hard dependency.
    """

    def __init__(self, bucket: str, prefix: str = "") -> None:
        self.bucket = bucket
        self.prefix = prefix

    def load(self) -> list[Document]:  # pragma: no cover - requires AWS credentials
        try:
            import boto3  # type: ignore
        except ImportError as exc:
            raise RuntimeError("S3Loader requires boto3: pip install boto3") from exc
        client = boto3.client("s3")
        docs: list[Document] = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                body = client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
                docs.append(
                    Document(
                        id=key,
                        text=body.decode("utf-8", errors="replace"),
                        metadata={"bucket": self.bucket, "key": key},
                    )
                )
        return docs
