from collections.abc import Sequence

import pytest
from corpus_engine import (
    DEFAULT_CHUNKERS,
    DEFAULT_PARSERS,
    ChunkerConfig,
    ChunkingStrategy,
    Document,
    DocumentElement,
    DocumentSource,
    ElementKind,
    create_default_chunker_registry,
    create_llm_chunker,
    create_semantic_chunker,
)
from corpus_engine.domain.errors import UnsupportedFormatError
from corpus_engine.llm.schemas import SemanticChunkGroup, SemanticChunkGroups
from pydantic import ValidationError


def source(filename: str) -> DocumentSource:
    return DocumentSource(id="document-1", uri=filename, filename=filename)


class FakeEmbedder:
    def __init__(
        self,
        vectors: list[list[float]],
        *,
        dimensions: int | None = None,
    ) -> None:
        self.vectors = vectors
        self._dimensions = dimensions if dimensions is not None else len(vectors[0])
        self.calls: list[list[str]] = []

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return self.vectors


def text_document(*contents: str) -> Document:
    document_source = source("notes.txt")
    elements = tuple(
        DocumentElement(
            id=f"element-{index}",
            document_id=document_source.id,
            kind=ElementKind.TEXT,
            content=content,
        )
        for index, content in enumerate(contents)
    )
    return Document(
        id=document_source.id,
        source=document_source,
        elements=elements,
    )


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


@pytest.mark.parametrize("threshold", [-1.01, 1.01])
def test_invalid_semantic_similarity_threshold_is_rejected(
    threshold: float,
) -> None:
    with pytest.raises(ValidationError):
        ChunkerConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            semantic_similarity_threshold=threshold,
        )


def test_semantic_strategy_is_not_claimed_without_an_adapter() -> None:
    capabilities = DEFAULT_CHUNKERS.capabilities()

    assert "semantic" not in capabilities
    assert "llm" in capabilities


@pytest.mark.parametrize(
    "vectors",
    [
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 1.0, 0.0]],
    ],
)
def test_semantic_chunker_is_model_dimension_agnostic(
    vectors: list[list[float]],
) -> None:
    document = text_document("Alpha one. Alpha two. Beta one.")
    embedder = FakeEmbedder(vectors)

    chunks = create_semantic_chunker(embedder)(
        document,
        ChunkerConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=100,
            semantic_similarity_threshold=0.8,
        ),
    )

    assert [chunk.content for chunk in chunks] == [
        "Alpha one. Alpha two.",
        "Beta one.",
    ]
    assert embedder.calls == [["Alpha one.", "Alpha two.", "Beta one."]]
    assert chunks[0].metadata["strategy"] == "semantic"
    assert chunks[0].metadata["sentence_indices"] == [0, 1]
    assert chunks[0].metadata["similarity_threshold"] == 0.8


def test_semantic_chunker_includes_similarity_threshold_boundary() -> None:
    document = text_document("First sentence. Second sentence.")
    embedder = FakeEmbedder([[1.0, 0.0], [0.5, 0.8660254037844386]])

    chunks = create_semantic_chunker(embedder)(
        document,
        ChunkerConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=100,
            semantic_similarity_threshold=0.5,
        ),
    )

    assert [chunk.content for chunk in chunks] == [
        "First sentence. Second sentence.",
    ]


def test_semantic_chunker_respects_size_and_splits_oversized_sentences() -> None:
    document = text_document("abcdefghijklmnopqrst.")
    embedder = FakeEmbedder([[1.0, 0.0]])

    chunks = create_semantic_chunker(embedder)(
        document,
        ChunkerConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=8,
            chunk_overlap=0,
            semantic_similarity_threshold=0.75,
        ),
    )

    assert [chunk.content for chunk in chunks] == ["abcdefgh", "ijklmnop", "qrst."]
    assert [chunk.metadata["semantic_part"] for chunk in chunks] == [0, 1, 2]
    assert all(chunk.metadata["sentence_indices"] == [0] for chunk in chunks)


def test_semantic_chunker_preserves_multi_element_provenance() -> None:
    document = text_document("First sentence.", "Related sentence.")
    embedder = FakeEmbedder([[1.0, 0.0], [1.0, 0.0]])

    chunks = create_semantic_chunker(embedder)(
        document,
        ChunkerConfig(
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=100,
        ),
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["element_ids"] == ["element-0", "element-1"]
    assert chunks[0].metadata["element_id"] == "element-0"


def test_semantic_chunker_skips_embedding_for_empty_documents() -> None:
    embedder = FakeEmbedder([[1.0, 0.0]])

    chunks = create_semantic_chunker(embedder)(
        text_document(),
        ChunkerConfig(strategy=ChunkingStrategy.SEMANTIC),
    )

    assert chunks == []
    assert embedder.calls == []


@pytest.mark.parametrize(
    ("embedder", "message"),
    [
        (
            FakeEmbedder([[1.0, 0.0]]),
            "exactly one vector per sentence",
        ),
        (
            FakeEmbedder([[1.0, 0.0], [1.0, 0.0]], dimensions=3),
            "inconsistent vector dimension",
        ),
        (
            FakeEmbedder([[1.0, 0.0], [0.0, 0.0]]),
            "zero vector",
        ),
    ],
)
def test_semantic_chunker_rejects_invalid_embeddings(
    embedder: FakeEmbedder,
    message: str,
) -> None:
    document = text_document("First sentence. Second sentence.")

    with pytest.raises(ValueError, match=message):
        create_semantic_chunker(embedder)(
            document,
            ChunkerConfig(strategy=ChunkingStrategy.SEMANTIC),
        )


def test_configured_registry_advertises_and_executes_semantic_chunking() -> None:
    embedder = FakeEmbedder([[1.0, 0.0], [1.0, 0.0]])
    registry = create_default_chunker_registry(embedder=embedder)
    document = text_document("First sentence. Related sentence.")

    chunks = registry.chunk(
        document,
        ChunkerConfig(strategy=ChunkingStrategy.SEMANTIC),
    )

    assert ChunkingStrategy.SEMANTIC in registry.capabilities()
    assert [chunk.content for chunk in chunks] == [
        "First sentence. Related sentence.",
    ]


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
