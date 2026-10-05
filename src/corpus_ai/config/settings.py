from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: SecretStr = Field(..., env="GEMINI_API_KEY")
    gemini_generation_model: str = "gemini-3.8-flash"
    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: Literal[3072] = 3072
    database_url: str = "postgresql+psycopg://corpus_user:corpus_password@localhost:5433/corpus_db"
    chunk_size: int = 1400
    chunk_overlap: int = 200
    retrieval_top_k: int = 5
    max_upload_mb: int = 25
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
