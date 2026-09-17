"""Optional language-model adapters."""

from corpus_engine.llm.client import StructuredInvoker, invoke_structured
from corpus_engine.llm.model_router import (
    ModelRegistry,
    ModelRouter,
    build_model_registry,
    get_model_router,
)
from corpus_engine.llm.schemas import SemanticChunkGroup, SemanticChunkGroups

__all__ = [
    "ModelRegistry",
    "ModelRouter",
    "SemanticChunkGroup",
    "SemanticChunkGroups",
    "StructuredInvoker",
    "build_model_registry",
    "get_model_router",
    "invoke_structured",
]
