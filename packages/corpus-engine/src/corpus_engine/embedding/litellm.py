"""LiteLLM-backed embedding adapter."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module
from typing import Any

from corpus_engine.config import Config, get_config
from corpus_engine.domain.errors import MissingDependencyError

EmbeddingInvoker = Callable[..., Any]


def _default_embedding_invoker(**kwargs: Any) -> Any:
    try:
        litellm = import_module("litellm")
    except ImportError as exc:
        raise MissingDependencyError(
            "Embedding dependency 'litellm' is not installed; "
            "install corpus-engine[litellm]",
        ) from exc
    return litellm.embedding(**kwargs)


class LiteLLMEmbedder:
    """Embed text through any LiteLLM-supported provider."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
        invoke: EmbeddingInvoker | None = None,
    ) -> None:
        if not model:
            raise ValueError("Embedding model cannot be empty")
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self._invoke = invoke or _default_embedding_invoker
        self._dimensions: int | None = None

    @classmethod
    def from_config(cls, config: Config | None = None) -> LiteLLMEmbedder:
        configuration = config or get_config()
        model = configuration.embedder_model
        if not model:
            raise RuntimeError(
                "No embedding model is configured; set CORPUS_EMBEDDER_MODEL",
            )
        return cls(
            model,
            api_key=configuration.llm_api_key,
            api_base=configuration.llm_api_base,
        )

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            raise RuntimeError("Embedding dimensions are unavailable before embed()")
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        parameters: dict[str, Any] = {
            "model": self.model,
            "input": list(texts),
        }
        if self.api_key:
            parameters["api_key"] = self.api_key
        if self.api_base:
            parameters["api_base"] = self.api_base

        response = self._invoke(**parameters)
        data = response["data"] if isinstance(response, dict) else response.data
        ordered = sorted(
            data,
            key=lambda item: (
                item.get("index", 0)
                if isinstance(item, dict)
                else getattr(item, "index", 0)
            ),
        )
        vectors = [
            [
                float(value)
                for value in (
                    item["embedding"]
                    if isinstance(item, dict)
                    else item.embedding
                )
            ]
            for item in ordered
        ]
        if not vectors:
            raise ValueError("Embedding provider returned no vectors")

        self._dimensions = len(vectors[0])
        return vectors
