import os
import pytest
from pydantic import ValidationError

from corpus_ai.config.settings import Settings, get_settings

pytestmark = pytest.mark.unit


def test_settings_default_values():
    """Test that Settings loads with correct default values"""
    os.environ["GEMINI_API_KEY"] = "test_key"
    settings = Settings()
    
    assert settings.gemini_api_key.get_secret_value() == "test_key"
    assert settings.gemini_generation_model == "gemini-3.8-flash"
    assert settings.gemini_embedding_model == "gemini-embedding-001"
    assert settings.embedding_dimensions == 768
    assert settings.database_url == "postgresql+psycopg://corpus_app:corpus_dev@localhost:5432/corpus_ai"
    assert settings.chunk_size == 1400
    assert settings.chunk_overlap == 200
    assert settings.retrieval_top_k == 5
    assert settings.max_upload_mb == 25
    assert settings.log_level == "INFO"


def test_settings_custom_values():
    """Test that Settings can be customized via environment variables"""
    os.environ["GEMINI_API_KEY"] = "custom_key"
    os.environ["GEMINI_GENERATION_MODEL"] = "gemini-2.5-flash"
    os.environ["GEMINI_EMBEDDING_MODEL"] = "text-embedding-004"
    os.environ["DATABASE_URL"] = "postgresql://user:pass@host:5432/db"
    os.environ["CHUNK_SIZE"] = "1000"
    os.environ["CHUNK_OVERLAP"] = "100"
    os.environ["RETRIEVAL_TOP_K"] = "10"
    os.environ["MAX_UPLOAD_MB"] = "50"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    settings = Settings()
    
    assert settings.gemini_api_key.get_secret_value() == "custom_key"
    assert settings.gemini_generation_model == "gemini-2.5-flash"
    assert settings.gemini_embedding_model == "text-embedding-004"
    assert settings.database_url == "postgresql://user:pass@host:5432/db"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 100
    assert settings.retrieval_top_k == 10
    assert settings.max_upload_mb == 50
    assert settings.log_level == "DEBUG"


def test_settings_missing_api_key():
    """Test that Settings raises error when GEMINI_API_KEY is missing"""
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]
    
    with pytest.raises(ValidationError):
        Settings()


def test_get_settings_caching():
    """Test that get_settings uses caching"""
    os.environ["GEMINI_API_KEY"] = "cached_key"
    
    settings1 = get_settings()
    settings2 = get_settings()
    
    assert settings1 is settings2


def test_settings_embedding_dimensions_validation():
    """Test that embedding_dimensions is validated correctly"""
    os.environ["GEMINI_API_KEY"] = "test_key"
    
    settings = Settings()
    assert settings.embedding_dimensions == 768
