"""Optional FastAPI adapter for Corpus Engine."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from corpus_engine.application.engine import CorpusEngine
from corpus_engine.domain.errors import MissingDependencyError, UnsupportedFormatError
from corpus_engine.domain.models import (
    ChunkerConfig,
    ChunkingStrategy,
    EngineCapabilities,
    ProcessingResult,
)


def create_router(engine: CorpusEngine | None = None) -> APIRouter:
    corpus = engine or CorpusEngine()
    router = APIRouter(prefix="/v1/engine", tags=["engine"])

    @router.get("/capabilities", response_model=EngineCapabilities)
    def capabilities() -> EngineCapabilities:
        return corpus.capabilities()

    @router.post(
        "/documents/preview",
        response_model=ProcessingResult,
        status_code=status.HTTP_200_OK,
    )
    async def preview_document(
        file: Annotated[UploadFile, File(description="Document to inspect")],
        strategy: Annotated[ChunkingStrategy | None, Form()] = None,
        chunk_size: Annotated[int, Form(ge=1)] = 512,
        chunk_overlap: Annotated[int, Form(ge=0)] = 0,
        group_by: Annotated[str | None, Form()] = None,
        preview_limit: Annotated[int, Form(ge=0)] = 5,
    ) -> ProcessingResult:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required",
            )

        data = await file.read()
        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document content cannot be empty",
            )

        chunking = None
        if strategy is not None:
            try:
                chunking = ChunkerConfig(
                    strategy=strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    group_by=group_by,
                )
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=str(exc),
                ) from exc

        try:
            result = corpus.process_bytes(
                data,
                filename=file.filename,
                chunking=chunking,
            )
        except UnsupportedFormatError as exc:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=str(exc),
            ) from exc
        except MissingDependencyError as exc:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=str(exc),
            ) from exc

        return result.model_copy(update={"chunks": result.chunks[:preview_limit]})

    return router
