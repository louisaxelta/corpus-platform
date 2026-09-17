"""Environment-backed Corpus Engine configuration."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration loaded from process variables and the repository `.env` file."""

    CORPUS_LLM_MODEL: str | None = None
    CORPUS_LLM_API_KEY: str | None = None
    CORPUS_LLM_API_BASE: str | None = None
    CORPUS_LLM_TEMPERATURE: float | None = Field(default=None, ge=0)
    CORPUS_LLM_MAX_TOKENS: int | None = Field(default=None, ge=1)

    # Compatibility with the environment names used by rag-pipelines.
    OPENROUTER_LLM_MODEL: str | None = None
    OPENROUTER_API_KEY: str | None = None
    OPENROUTER_API_URL: str = "https://openrouter.ai/api/v1"
    MODEL_TEMPERATURE: float = Field(default=0.1, ge=0)
    MODEL_MAX_TOKENS: int = Field(default=16384, ge=1)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def llm_model(self) -> str | None:
        model = self.CORPUS_LLM_MODEL or self.OPENROUTER_LLM_MODEL
        if self.CORPUS_LLM_MODEL or not model or model.startswith("openrouter/"):
            return model
        return f"openrouter/{model}"

    @property
    def llm_api_key(self) -> str | None:
        return self.CORPUS_LLM_API_KEY or self.OPENROUTER_API_KEY

    @property
    def llm_api_base(self) -> str | None:
        if self.CORPUS_LLM_API_BASE:
            return self.CORPUS_LLM_API_BASE.rstrip("/")
        if self.OPENROUTER_LLM_MODEL:
            return self.OPENROUTER_API_URL.rstrip("/")
        return None

    @property
    def llm_temperature(self) -> float:
        return (
            self.CORPUS_LLM_TEMPERATURE
            if self.CORPUS_LLM_TEMPERATURE is not None
            else self.MODEL_TEMPERATURE
        )

    @property
    def llm_max_tokens(self) -> int:
        return self.CORPUS_LLM_MAX_TOKENS or self.MODEL_MAX_TOKENS


@lru_cache
def get_config() -> Config:
    """Return the process-wide environment configuration."""
    return Config()
