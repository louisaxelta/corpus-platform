"""Command-line adapter for local Corpus Engine processing."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from corpus_engine.application.engine import CorpusEngine
from corpus_engine.domain.models import ChunkerConfig, ChunkingStrategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="corpus-engine",
        description="Parse and chunk a local document without cloud services.",
    )
    parser.add_argument("document", help="Path to a supported local document")
    parser.add_argument(
        "--strategy",
        choices=[strategy.value for strategy in ChunkingStrategy],
        help="Chunking strategy; defaults according to the document format",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=0)
    parser.add_argument("--group-by")
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=5,
        help="Number of chunks to include in output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chunking = None
    if args.strategy:
        chunking = ChunkerConfig(
            strategy=ChunkingStrategy(args.strategy),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            group_by=args.group_by,
        )

    result = CorpusEngine().process_file(args.document, chunking=chunking)
    limit = max(args.preview_limit, 0)
    output = {
        "document_id": result.document.id,
        "filename": result.document.source.filename,
        "parser_format": result.parser_format,
        "chunking": result.chunking.model_dump(mode="json"),
        "element_count": len(result.document.elements),
        "chunk_count": len(result.chunks),
        "chunks": [
            chunk.model_dump(mode="json")
            for chunk in result.chunks[:limit]
        ],
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0
