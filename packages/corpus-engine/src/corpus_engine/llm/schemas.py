"""Structured responses used by LLM-backed chunking."""

from pydantic import BaseModel, ConfigDict, Field


class SemanticChunkGroup(BaseModel):
    """Ordered sentence indices belonging to one chunk."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    indices: list[int] = Field(
        min_length=1,
        description="Sentence indices that belong to the same semantic chunk",
    )


class SemanticChunkGroups(BaseModel):
    """Complete grouping returned by the language model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    groups: list[SemanticChunkGroup] = Field(
        description="Semantic groupings of sentence indices",
    )
