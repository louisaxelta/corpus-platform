import json
from pathlib import Path

import pytest
from corpus_engine import ChunkingStrategy, CorpusEngine
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
