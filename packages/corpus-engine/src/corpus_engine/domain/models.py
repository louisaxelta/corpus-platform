"""Versioned provider-neutral domain models."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CorpusModel(BaseModel):
    """Base model for stable public engine contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ElementKind(StrEnum):
    TEXT = "text"
    TABLE_ROW = "table_row"
    SECTION = "section"
    SLIDE = "slide"


class ChunkingStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SLIDING_WINDOW = "sliding_window"
    ROW = "row"
    SECTION = "section"
    FAQ = "faq"


class SourceLocation(CorpusModel):
    page: int | None = Field(default=None, ge=1)
    sheet: str | None = None
    slide: int | None = Field(default=None, ge=1)
    heading: str | None = None
    row: int | None = Field(default=None, ge=1)


class DocumentSource(CorpusModel):
    id: str
    uri: str
    filename: str
    media_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentElement(CorpusModel):
    id: str
    document_id: str
    kind: ElementKind
    content: str | dict[str, Any]
    location: SourceLocation = Field(default_factory=SourceLocation)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(CorpusModel):
    id: str
    source: DocumentSource
    elements: tuple[DocumentElement, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkerConfig(CorpusModel):
    strategy: ChunkingStrategy
    chunk_size: int = Field(default=512, ge=1)
    chunk_overlap: int = Field(default=50, ge=0)
    group_by: str | None = None

    @model_validator(mode="after")
    def overlap_must_be_smaller_than_size(self) -> ChunkerConfig:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class Chunk(CorpusModel):
    id: str
    document_id: str
    content: str
    ordinal: int = Field(ge=0)
    location: SourceLocation = Field(default_factory=SourceLocation)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionPlan(CorpusModel):
    schema_version: str = "1"
    parser: str
    chunking: ChunkerConfig
    embedder: str | None = None
    index: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestionResult(CorpusModel):
    document_id: str
    chunk_ids: tuple[str, ...]
    indexed_count: int = Field(ge=0)


class IndexRecord(CorpusModel):
    id: str
    document_id: str
    content: str
    vector: tuple[float, ...]
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResult(CorpusModel):
    record: IndexRecord
    score: float


class EngineCapabilities(CorpusModel):
    parser_formats: tuple[str, ...]
    chunking_strategies: tuple[ChunkingStrategy, ...]


class ProcessingResult(CorpusModel):
    document: Document
    chunks: tuple[Chunk, ...]
    parser_format: str
    chunking: ChunkerConfig
