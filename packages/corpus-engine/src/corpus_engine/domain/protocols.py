"""Adapter contracts implemented by local and vendor integrations."""

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

from corpus_engine.domain.models import (
    Chunk,
    ChunkerConfig,
    Document,
    DocumentElement,
    DocumentSource,
    IndexRecord,
    QueryResult,
)


@runtime_checkable
class SourceReader(Protocol):
    def read(self, source: DocumentSource) -> bytes: ...


@runtime_checkable
class Parser(Protocol):
    def parse(self, data: bytes, source: DocumentSource) -> list[DocumentElement]: ...


@runtime_checkable
class Chunker(Protocol):
    def chunk(
        self,
        document: Document,
        config: ChunkerConfig,
    ) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class Index(Protocol):
    def upsert(self, records: Iterable[IndexRecord]) -> None: ...

    def delete(self, ids: Iterable[str]) -> None: ...

    def query(
        self,
        vector: Sequence[float],
        *,
        limit: int = 10,
        filters: dict[str, object] | None = None,
    ) -> list[QueryResult]: ...


@runtime_checkable
class ObjectStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...

    def get(self, key: str) -> bytes: ...

    def delete(self, key: str) -> None: ...
