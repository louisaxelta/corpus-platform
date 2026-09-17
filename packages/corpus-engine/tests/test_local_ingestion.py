import pytest
from corpus_engine import (
    DEFAULT_CHUNKERS,
    DEFAULT_PARSERS,
    ChunkerConfig,
    ChunkingStrategy,
    Document,
    DocumentSource,
    ElementKind,
    create_llm_chunker,
)
from corpus_engine.domain.errors import UnsupportedFormatError
from corpus_engine.llm.schemas import SemanticChunkGroup, SemanticChunkGroups
from pydantic import ValidationError


def source(filename: str) -> DocumentSource:
    return DocumentSource(id="document-1", uri=filename, filename=filename)


def test_text_parser_is_distinct_from_pdf_parser() -> None:
    parser = DEFAULT_PARSERS.create_for("notes.txt")

    elements = parser.parse(b"Local text content", source("notes.txt"))

    assert len(elements) == 1
    assert elements[0].kind is ElementKind.TEXT
    assert elements[0].content == "Local text content"


def test_csv_parser_preserves_row_location() -> None:
    parser = DEFAULT_PARSERS.create_for("faq.csv")

    elements = parser.parse(
        b"question,answer\nWhat is Corpus?,A document engine\n",
        source("faq.csv"),
    )

    assert elements[0].kind is ElementKind.TABLE_ROW
    assert elements[0].location.row == 2


def test_legacy_binary_formats_are_not_advertised() -> None:
    with pytest.raises(UnsupportedFormatError):
        DEFAULT_PARSERS.create_for("legacy.doc")


def test_sliding_chunks_overlap_and_keep_provenance() -> None:
    document_source = source("notes.txt")
    elements = DEFAULT_PARSERS.create_for("notes.txt").parse(
        b"abcdefghij",
        document_source,
    )
    document = Document(
        id=document_source.id,
        source=document_source,
        elements=tuple(elements),
    )

    chunks = DEFAULT_CHUNKERS.chunk(
        document,
        ChunkerConfig(
            strategy=ChunkingStrategy.SLIDING_WINDOW,
            chunk_size=5,
            chunk_overlap=2,
        ),
    )

    assert [chunk.content for chunk in chunks] == ["abcde", "defgh", "ghij", "j"]
    assert all(chunk.metadata["element_id"] == elements[0].id for chunk in chunks)


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChunkerConfig(
            strategy=ChunkingStrategy.FIXED,
            chunk_size=10,
            chunk_overlap=10,
        )


def test_semantic_strategy_is_not_claimed_without_an_adapter() -> None:
    capabilities = DEFAULT_CHUNKERS.capabilities()

    assert "semantic" not in capabilities
    assert "llm" in capabilities


def test_llm_chunker_groups_sentences_and_preserves_provenance() -> None:
    document_source = source("notes.txt")
    elements = DEFAULT_PARSERS.create_for("notes.txt").parse(
        (
            b"Authentication starts here. Coffee is unrelated. "
            b"Authentication continues here."
        ),
        document_source,
    )
    document = Document(
        id=document_source.id,
        source=document_source,
        elements=tuple(elements),
    )

    def invoke(
        prompt: str,
        response_model: type[SemanticChunkGroups],
        max_tokens: int | None,
    ) -> SemanticChunkGroups:
        assert "[0] Authentication starts here." in prompt
        assert max_tokens == 16384
        return response_model(
            groups=[
                SemanticChunkGroup(indices=[0, 2]),
                SemanticChunkGroup(indices=[1]),
            ],
        )

    chunks = create_llm_chunker(invoke)(
        document,
        ChunkerConfig(strategy=ChunkingStrategy.LLM, chunk_size=100),
    )

    assert [chunk.content for chunk in chunks] == [
        "Authentication starts here. Authentication continues here.",
        "Coffee is unrelated.",
    ]
    assert chunks[0].metadata["element_ids"] == [elements[0].id]
    assert chunks[0].metadata["sentence_indices"] == [0, 2]


def test_llm_chunker_splits_one_topic_at_the_size_limit() -> None:
    document_source = source("notes.txt")
    elements = DEFAULT_PARSERS.create_for("notes.txt").parse(
        b"First topic sentence. Second topic sentence.",
        document_source,
    )
    document = Document(
        id=document_source.id,
        source=document_source,
        elements=tuple(elements),
    )

    def invoke(
        prompt: str,
        response_model: type[SemanticChunkGroups],
        max_tokens: int | None,
    ) -> SemanticChunkGroups:
        return response_model(
            groups=[SemanticChunkGroup(indices=[0, 1])],
        )

    chunks = create_llm_chunker(invoke)(
        document,
        ChunkerConfig(
            strategy=ChunkingStrategy.LLM,
            chunk_size=25,
            chunk_overlap=0,
        ),
    )

    assert [chunk.content for chunk in chunks] == [
        "First topic sentence.",
        "Second topic sentence.",
    ]
    assert [chunk.metadata["topic_group"] for chunk in chunks] == [0, 0]
    assert [chunk.metadata["topic_part"] for chunk in chunks] == [0, 1]
