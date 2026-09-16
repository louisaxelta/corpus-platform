"""Small client facade used by LLM-backed engine adapters."""

from typing import Protocol, TypeVar

from pydantic import BaseModel

from corpus_engine.llm.model_router import get_model_router

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class StructuredInvoker(Protocol):
    """Callable contract that makes LLM chunking testable and replaceable."""

    def __call__(
        self,
        prompt: str,
        response_model: type[ResponseT],
        max_tokens: int | None = None,
    ) -> ResponseT: ...


def invoke_structured(
    prompt: str,
    response_model: type[ResponseT],
    max_tokens: int | None = None,
) -> ResponseT:
    """Invoke the default routed model with Pydantic validation."""
    messages = [{"role": "user", "content": prompt}]
    if max_tokens is None:
        return get_model_router().structured(
            messages=messages,
            response_model=response_model,
        )
    return get_model_router().structured(
        messages=messages,
        response_model=response_model,
        max_tokens=max_tokens,
    )
