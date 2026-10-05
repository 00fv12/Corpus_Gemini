from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: SecretStr
    gemini_generation_model: str = "gemini-3.8-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: Literal[768] = 768
    database_url: str = "postgresql+psycopg://corpus_app:corpus_dev@localhost:5432/corpus_ai"
    chunk_size: int = 1400
    chunk_overlap: int = 200
    retrieval_top_k: int = 5
    max_upload_mb: int = 25
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
