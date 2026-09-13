"""Provider-neutral document intelligence engine."""

from corpus_engine.application.engine import CorpusEngine
from corpus_engine.domain.models import (
    Chunk,
    ChunkerConfig,
    ChunkingStrategy,
    Document,
    DocumentElement,
    DocumentSource,
    ElementKind,
    EngineCapabilities,
    IngestionPlan,
    IngestionResult,
    ProcessingResult,
    SourceLocation,
)
from corpus_engine.ingestion.chunkers import DEFAULT_CHUNKERS, ChunkerRegistry
from corpus_engine.ingestion.parsers import DEFAULT_PARSERS, ParserRegistry

__version__ = "0.1.0a0"

__all__ = [
    "DEFAULT_CHUNKERS",
    "DEFAULT_PARSERS",
    "Chunk",
    "ChunkerConfig",
    "ChunkerRegistry",
    "ChunkingStrategy",
    "CorpusEngine",
    "Document",
    "DocumentElement",
    "DocumentSource",
    "ElementKind",
    "EngineCapabilities",
    "IngestionPlan",
    "IngestionResult",
    "ParserRegistry",
    "ProcessingResult",
    "SourceLocation",
]
