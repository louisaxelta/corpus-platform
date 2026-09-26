"""Environment-backed Corpus Engine configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration loaded from process variables and the repository `.env` file."""

    CORPUS_LLM_MODEL: str | None = None
    CORPUS_EMBEDDER_MODEL: str | None = None
    CORPUS_LLM_API_KEY: str | None = None
    CORPUS_LLM_API_BASE: str | None = None
    CORPUS_LLM_TEMPERATURE: float = Field(default=0.1, ge=0)
    CORPUS_LLM_MAX_TOKENS: int = Field(default=16384, ge=1)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def llm_model(self) -> str | None:
        return self.CORPUS_LLM_MODEL

    @property
    def embedder_model(self) -> str | None:
        model = self.CORPUS_EMBEDDER_MODEL
        if not model or model.startswith("openrouter/"):
            return model
        return f"openrouter/{model}"

    @property
    def llm_api_key(self) -> str | None:
        return self.CORPUS_LLM_API_KEY

    @property
    def llm_api_base(self) -> str | None:
        return self.CORPUS_LLM_API_BASE.rstrip("/") if self.CORPUS_LLM_API_BASE else None

    @property
    def llm_temperature(self) -> float:
        return self.CORPUS_LLM_TEMPERATURE

    @property
    def llm_max_tokens(self) -> int:
        return self.CORPUS_LLM_MAX_TOKENS


@lru_cache
def get_config() -> Config:
    """Return the process-wide environment configuration."""
    return Config()
