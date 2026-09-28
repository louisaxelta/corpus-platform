"""Provider-neutral document intelligence engine."""

from corpus_engine.application.engine import CorpusEngine
from corpus_engine.config import Config, get_config
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
from corpus_engine.embedding import LiteLLMEmbedder
from corpus_engine.ingestion.chunkers import (
    DEFAULT_CHUNKERS,
    ChunkerRegistry,
    create_default_chunker_registry,
    create_llm_chunker,
    create_semantic_chunker,
)
from corpus_engine.ingestion.parsers import DEFAULT_PARSERS, ParserRegistry

__version__ = "0.1.0a0"

__all__ = [
    "DEFAULT_CHUNKERS",
    "DEFAULT_PARSERS",
    "Chunk",
    "ChunkerConfig",
    "ChunkerRegistry",
    "ChunkingStrategy",
    "Config",
    "CorpusEngine",
    "Document",
    "DocumentElement",
    "DocumentSource",
    "ElementKind",
    "EngineCapabilities",
    "IngestionPlan",
    "IngestionResult",
    "LiteLLMEmbedder",
    "ParserRegistry",
    "ProcessingResult",
    "SourceLocation",
    "create_default_chunker_registry",
    "create_llm_chunker",
    "create_semantic_chunker",
    "get_config",
]
