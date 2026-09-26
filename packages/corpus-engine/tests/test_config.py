from corpus_engine.config import get_config
from corpus_engine.llm.model_router import build_model_registry
from pytest import MonkeyPatch


def test_llm_config_is_loaded_from_environment(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("CORPUS_LLM_MODEL", "openrouter/example/model")
    monkeypatch.setenv("CORPUS_LLM_API_KEY", "test-key")
    monkeypatch.setenv("CORPUS_LLM_API_BASE", "https://example.test/v1/")
    monkeypatch.setenv("CORPUS_LLM_TEMPERATURE", "0.25")
    monkeypatch.setenv("CORPUS_LLM_MAX_TOKENS", "2048")
    get_config.cache_clear()

    config = get_config()
    registry, default_model = build_model_registry(config)

    assert default_model == "primary"
    assert registry["primary"] == {
        "model": "openrouter/example/model",
        "api_key": "test-key",
        "api_base": "https://example.test/v1",
    }
    assert config.llm_temperature == 0.25
    assert config.llm_max_tokens == 2048
    get_config.cache_clear()


def test_embedder_config_reuses_llm_credentials(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv("CORPUS_EMBEDDER_MODEL", "qwen/qwen3-embedding-8b")
    monkeypatch.setenv("CORPUS_LLM_API_KEY", "shared-key")
    monkeypatch.setenv("CORPUS_LLM_API_BASE", "https://openrouter.test/api/v1/")
    get_config.cache_clear()

    config = get_config()

    assert config.embedder_model == "openrouter/qwen/qwen3-embedding-8b"
    assert config.llm_api_key == "shared-key"
    assert config.llm_api_base == "https://openrouter.test/api/v1"
    get_config.cache_clear()
