"""Primary library entry point for local document processing."""

from __future__ import annotations

import hashlib
from pathlib import Path

from corpus_engine.domain.models import (
    ChunkerConfig,
    ChunkingStrategy,
    Document,
    DocumentSource,
    EngineCapabilities,
    ProcessingResult,
)
from corpus_engine.ingestion.chunkers import DEFAULT_CHUNKERS, ChunkerRegistry
from corpus_engine.ingestion.parsers import DEFAULT_PARSERS, ParserRegistry

_TABULAR_FORMATS = {".csv", ".xlsx"}
_STRUCTURAL_FORMATS = {".docx", ".pptx"}


class CorpusEngine:
    """Coordinates provider-neutral parsing and chunking.

    Embedding and indexing are injected adapter stages in a later pipeline layer;
    this entry point intentionally performs no network or cloud operations.
    """

    def __init__(
        self,
        *,
        parsers: ParserRegistry | None = None,
        chunkers: ChunkerRegistry | None = None,
    ) -> None:
        self.parsers = parsers or DEFAULT_PARSERS
        self.chunkers = chunkers or DEFAULT_CHUNKERS

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            parser_formats=self.parsers.capabilities(),
            chunking_strategies=self.chunkers.capabilities(),
        )

    def process_file(
        self,
        path: str | Path,
        *,
        chunking: ChunkerConfig | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ProcessingResult:
        file_path = Path(path)
        data = file_path.read_bytes()
        return self.process_bytes(
            data,
            filename=file_path.name,
            uri=str(file_path.resolve()),
            chunking=chunking,
            metadata=metadata,
        )

    def process_bytes(
        self,
        data: bytes,
        *,
        filename: str,
        uri: str | None = None,
        chunking: ChunkerConfig | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ProcessingResult:
        if not data:
            raise ValueError("Document content cannot be empty")

        extension = Path(filename).suffix.lower()
        parser = self.parsers.create_for(filename)
        document_id = hashlib.sha256(
            filename.encode() + b"\0" + data,
        ).hexdigest()[:24]
        source = DocumentSource(
            id=document_id,
            uri=uri or filename,
            filename=filename,
            metadata=dict(metadata or {}),
        )
        elements = parser.parse(data, source)
        document = Document(
            id=document_id,
            source=source,
            elements=tuple(elements),
        )
        effective_chunking = chunking or self.default_chunking_for(extension)
        chunks = self.chunkers.chunk(document, effective_chunking)
        return ProcessingResult(
            document=document,
            chunks=tuple(chunks),
            parser_format=extension,
            chunking=effective_chunking,
        )

    @staticmethod
    def default_chunking_for(extension: str) -> ChunkerConfig:
        normalized = extension.lower()
        if normalized in _TABULAR_FORMATS:
            return ChunkerConfig(
                strategy=ChunkingStrategy.ROW,
                chunk_size=1,
                chunk_overlap=0,
            )
        if normalized in _STRUCTURAL_FORMATS:
            return ChunkerConfig(
                strategy=ChunkingStrategy.SECTION,
                chunk_size=512,
                chunk_overlap=50,
            )
        return ChunkerConfig(
            strategy=ChunkingStrategy.FIXED,
            chunk_size=512,
            chunk_overlap=0,
        )
