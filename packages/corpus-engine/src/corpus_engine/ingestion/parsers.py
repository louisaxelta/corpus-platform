"""Local document parsers and extension-based parser discovery."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable
from importlib import import_module
from io import BytesIO
from pathlib import Path
from typing import Any

from corpus_engine.domain.errors import MissingDependencyError, UnsupportedFormatError
from corpus_engine.domain.models import (
    DocumentElement,
    DocumentSource,
    ElementKind,
    SourceLocation,
)
from corpus_engine.domain.protocols import Parser

ParserFactory = Callable[[], Parser]


def _element(
    source: DocumentSource,
    kind: ElementKind,
    content: str | dict[str, Any],
    ordinal: int,
    location: SourceLocation | None = None,
) -> DocumentElement:
    serialized = json.dumps(content, sort_keys=True, default=str)
    digest = hashlib.sha256(
        f"{source.id}:{kind}:{ordinal}:{serialized}".encode(),
    ).hexdigest()[:24]
    return DocumentElement(
        id=digest,
        document_id=source.id,
        kind=kind,
        content=content,
        location=location or SourceLocation(),
    )


def _optional_module(name: str) -> Any:
    try:
        return import_module(name)
    except ImportError as exc:
        raise MissingDependencyError(
            f"Parser dependency '{name}' is not installed; "
            "install corpus-engine[parsers]",
        ) from exc


class TextParser:
    def parse(self, data: bytes, source: DocumentSource) -> list[DocumentElement]:
        text = data.decode("utf-8-sig").strip()
        return [_element(source, ElementKind.TEXT, text, 0)] if text else []


class MarkdownParser(TextParser):
    pass


class CsvParser:
    def parse(self, data: bytes, source: DocumentSource) -> list[DocumentElement]:
        text = data.decode("utf-8-sig")
        rows = csv.DictReader(text.splitlines())
        elements: list[DocumentElement] = []
        for row_number, row in enumerate(rows, start=2):
            if not any(value for value in row.values()):
                continue
            elements.append(
                _element(
                    source,
                    ElementKind.TABLE_ROW,
                    dict(row),
                    len(elements),
                    SourceLocation(row=row_number),
                ),
            )
        return elements


class PdfParser:
    def parse(self, data: bytes, source: DocumentSource) -> list[DocumentElement]:
        pypdf = _optional_module("pypdf")
        reader = pypdf.PdfReader(BytesIO(data))
        elements: list[DocumentElement] = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                elements.append(
                    _element(
                        source,
                        ElementKind.TEXT,
                        text,
                        len(elements),
                        SourceLocation(page=page_number),
                    ),
                )
        return elements


class ExcelParser:
    def parse(self, data: bytes, source: DocumentSource) -> list[DocumentElement]:
        openpyxl = _optional_module("openpyxl")
        workbook = openpyxl.load_workbook(BytesIO(data), data_only=True, read_only=True)
        elements: list[DocumentElement] = []
        for worksheet in workbook.worksheets:
            rows = worksheet.iter_rows(values_only=True)
            headers = next(rows, None)
            if headers is None:
                continue
            names = [
                str(value).strip() if value is not None else f"column_{index + 1}"
                for index, value in enumerate(headers)
            ]
            for row_number, row in enumerate(rows, start=2):
                if all(value is None for value in row):
                    continue
                content = {
                    name: row[index] if index < len(row) else None
                    for index, name in enumerate(names)
                }
                elements.append(
                    _element(
                        source,
                        ElementKind.TABLE_ROW,
                        content,
                        len(elements),
                        SourceLocation(sheet=worksheet.title, row=row_number),
                    ),
                )
        return elements


class WordParser:
    def parse(self, data: bytes, source: DocumentSource) -> list[DocumentElement]:
        document = _optional_module("docx").Document(BytesIO(data))
        elements: list[DocumentElement] = []
        heading: str | None = None
        body: list[str] = []

        def flush() -> None:
            if not body:
                return
            content = {"heading": heading, "body": " ".join(body)}
            elements.append(
                _element(
                    source,
                    ElementKind.SECTION,
                    content,
                    len(elements),
                    SourceLocation(heading=heading),
                ),
            )
            body.clear()

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()
            if not text:
                continue
            if paragraph.style.name.startswith("Heading"):
                flush()
                heading = text
            else:
                body.append(text)
        flush()
        return elements


class PowerPointParser:
    def parse(self, data: bytes, source: DocumentSource) -> list[DocumentElement]:
        presentation = _optional_module("pptx").Presentation(BytesIO(data))
        elements: list[DocumentElement] = []
        for slide_number, slide in enumerate(presentation.slides, start=1):
            title_shape = slide.shapes.title
            title = title_shape.text.strip() if title_shape is not None else None
            body = [
                shape.text.strip()
                for shape in slide.shapes
                if shape is not title_shape
                and getattr(shape, "has_text_frame", False)
                and shape.text.strip()
            ]
            if title or body:
                elements.append(
                    _element(
                        source,
                        ElementKind.SLIDE,
                        {"title": title, "body": " ".join(body)},
                        len(elements),
                        SourceLocation(slide=slide_number),
                    ),
                )
        return elements


class ParserRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ParserFactory] = {}

    def register(self, extension: str, factory: ParserFactory) -> None:
        normalized = extension.lower()
        if not normalized.startswith("."):
            normalized = f".{normalized}"
        self._factories[normalized] = factory

    def create_for(self, filename: str) -> Parser:
        extension = Path(filename).suffix.lower()
        try:
            return self._factories[extension]()
        except KeyError as exc:
            supported = ", ".join(sorted(self._factories))
            raise UnsupportedFormatError(
                f"Unsupported document format '{extension}'; supported: {supported}",
            ) from exc

    def capabilities(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


DEFAULT_PARSERS = ParserRegistry()
DEFAULT_PARSERS.register(".txt", TextParser)
DEFAULT_PARSERS.register(".md", MarkdownParser)
DEFAULT_PARSERS.register(".csv", CsvParser)
DEFAULT_PARSERS.register(".pdf", PdfParser)
DEFAULT_PARSERS.register(".xlsx", ExcelParser)
DEFAULT_PARSERS.register(".docx", WordParser)
DEFAULT_PARSERS.register(".pptx", PowerPointParser)
