"""Optional LiteLLM router with structured-output validation."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Any, TypeVar

from pydantic import BaseModel

from corpus_engine.config import Config, get_config
from corpus_engine.domain.errors import MissingDependencyError

ResponseT = TypeVar("ResponseT", bound=BaseModel)
ModelRegistry = dict[str, dict[str, Any]]


def build_model_registry(config: Config | None = None) -> tuple[ModelRegistry, str]:
    """Build a provider-neutral registry from environment configuration."""
    configuration = config or get_config()
    model = configuration.llm_model
    if not model:
        raise RuntimeError(
            "No LLM model is configured; set CORPUS_LLM_MODEL "
            "(for example, 'openrouter/anthropic/claude-sonnet-4').",
        )

    parameters: dict[str, Any] = {"model": model}
    if configuration.llm_api_key:
        parameters["api_key"] = configuration.llm_api_key
    if configuration.llm_api_base:
        parameters["api_base"] = configuration.llm_api_base
    return {"primary": parameters}, "primary"


def _optional_module(name: str) -> Any:
    try:
        return import_module(name)
    except ImportError as exc:
        raise MissingDependencyError(
            f"LLM dependency '{name}' is not installed; install corpus-engine[litellm]",
        ) from exc


class ModelRouter:
    """LiteLLM model registry with optional fallback and Instructor validation."""

    def __init__(
        self,
        registry: ModelRegistry | None = None,
        default_model: str | None = None,
        fallback_models: list[str] | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        config = get_config()
        if registry is None:
            registry, configured_default = build_model_registry(config)
            default_model = default_model or configured_default
        if not registry:
            raise ValueError("Model registry cannot be empty")

        self.registry = registry
        self.default_model = default_model or next(iter(registry))
        self.fallback_models = fallback_models or []
        self.temperature = config.llm_temperature if temperature is None else temperature
        self.max_tokens = config.llm_max_tokens if max_tokens is None else max_tokens

        if self.default_model not in registry:
            raise ValueError(f"Default model '{self.default_model}' is not registered")
        unknown_fallbacks = set(self.fallback_models) - registry.keys()
        if unknown_fallbacks:
            raise ValueError(
                f"Fallback models are not registered: {sorted(unknown_fallbacks)}",
            )

        litellm = _optional_module("litellm")
        instructor = _optional_module("instructor")
        litellm.drop_params = True
        model_list = [
            {"model_name": name, "litellm_params": parameters}
            for name, parameters in registry.items()
        ]
        fallbacks = (
            [{self.default_model: self.fallback_models}]
            if self.fallback_models
            else []
        )
        self._router = litellm.Router(
            model_list=model_list,
            fallbacks=fallbacks,
            num_retries=2,
        )
        self._instructor = instructor.from_litellm(self._completion)

    def _completion(self, **kwargs: Any) -> Any:
        """Restore deployment credentials before every Instructor retry."""
        resolved = str(kwargs.get("model") or self.default_model)
        deployment = self.registry.get(resolved)
        if deployment:
            if deployment.get("api_key"):
                kwargs["api_key"] = deployment["api_key"]
            if deployment.get("api_base"):
                kwargs["api_base"] = deployment["api_base"]
        return self._router.completion(**kwargs)

    def structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[ResponseT],
        model: str | None = None,
        max_retries: int = 2,
        **kwargs: Any,
    ) -> ResponseT:
        """Return output validated against a Pydantic model."""
        resolved = model or self.default_model
        if resolved not in self.registry:
            raise ValueError(f"Model '{resolved}' is not registered")
        deployment = self.registry[resolved]
        if deployment.get("api_key") and kwargs.get("api_key") is None:
            kwargs["api_key"] = deployment["api_key"]
        if deployment.get("api_base") and kwargs.get("api_base") is None:
            kwargs["api_base"] = deployment["api_base"]
        kwargs.setdefault("temperature", self.temperature)
        kwargs.setdefault("max_tokens", self.max_tokens)
        result: ResponseT = self._instructor.chat.completions.create(
            model=resolved,
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            **kwargs,
        )
        return result


@lru_cache
def get_model_router() -> ModelRouter:
    """Return the lazily configured default model router."""
    return ModelRouter()
