import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from corpus_engine import ChunkingStrategy, CorpusEngine
from corpus_engine.embedding import LiteLLMEmbedder
from corpus_engine.interfaces.cli import main


def test_engine_processes_bytes_with_format_defaults() -> None:
    result = CorpusEngine().process_bytes(
        b"question,answer\nWhat is Corpus?,A document engine\n",
        filename="faq.csv",
    )

    assert result.chunking.strategy is ChunkingStrategy.ROW
    assert len(result.document.elements) == 1
    assert len(result.chunks) == 1


def test_document_ids_are_stable() -> None:
    engine = CorpusEngine()

    first = engine.process_bytes(b"same content", filename="notes.txt")
    second = engine.process_bytes(b"same content", filename="notes.txt")

    assert first.document.id == second.document.id
    assert first.chunks[0].id == second.chunks[0].id


def test_cli_is_a_local_entry_point(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    document = tmp_path / "notes.txt"
    document.write_text("Corpus runs locally.", encoding="utf-8")

    exit_code = main([str(document), "--preview-limit", "1"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["filename"] == "notes.txt"
    assert output["chunk_count"] == 1


class FakeCLIEmbedder:
    @property
    def dimensions(self) -> int:
        return 2

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_cli_configures_embedder_for_semantic_chunking(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = tmp_path / "notes.txt"
    document.write_text(
        "Corpus handles documents. Corpus chunks documents.",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        LiteLLMEmbedder,
        "from_config",
        classmethod(lambda cls: FakeCLIEmbedder()),
    )

    exit_code = main([str(document), "--strategy", "semantic"])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["chunking"]["strategy"] == "semantic"
    assert output["chunk_count"] == 1
