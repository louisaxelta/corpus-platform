from typing import Any

from corpus_engine.config import Config
from corpus_engine.embedding import LiteLLMEmbedder


def test_litellm_embedder_preserves_input_order_and_infers_dimensions() -> None:
    captured: dict[str, Any] = {}

    def invoke(**kwargs: Any) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0, 0.0]},
                {"index": 0, "embedding": [1.0, 0.0, 0.0]},
            ],
        }

    embedder = LiteLLMEmbedder(
        "openrouter/example/embedding-model",
        api_key="test-key",
        api_base="https://example.test/v1",
        invoke=invoke,
    )

    vectors = embedder.embed(["first", "second"])

    assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert embedder.dimensions == 3
    assert captured == {
        "model": "openrouter/example/embedding-model",
        "input": ["first", "second"],
        "api_key": "test-key",
        "api_base": "https://example.test/v1",
    }


def test_litellm_embedder_rejects_empty_provider_response() -> None:
    embedder = LiteLLMEmbedder(
        "openrouter/example/embedding-model",
        invoke=lambda **kwargs: {"data": []},
    )

    try:
        embedder.embed(["text"])
    except ValueError as exc:
        assert str(exc) == "Embedding provider returned no vectors"
    else:
        raise AssertionError("Expected an empty embedding response to fail")


def test_litellm_embedder_reuses_llm_connection_config() -> None:
    config = Config(
        CORPUS_EMBEDDER_MODEL="openrouter/example/embedding-model",
        CORPUS_LLM_API_KEY="shared-key",
        CORPUS_LLM_API_BASE="https://openrouter.test/api/v1/",
    )

    embedder = LiteLLMEmbedder.from_config(config)

    assert embedder.api_key == "shared-key"
    assert embedder.api_base == "https://openrouter.test/api/v1"
