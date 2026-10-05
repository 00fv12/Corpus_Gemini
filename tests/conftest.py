"""Shared fixtures for tests."""
import os
import pytest
from unittest.mock import Mock, MagicMock
from pydantic import SecretStr

from corpus_ai.config.settings import Settings
from corpus_ai.providers.gemini import GeminiProvider
from corpus_ai.storage.repository import CorpusRepository
from corpus_ai.storage.models import CorpusChunk


@pytest.fixture
def mock_settings():
    """Mock Settings for testing with default values."""
    settings = Mock(spec=Settings)
    settings.gemini_api_key = SecretStr("test_api_key")
    settings.gemini_generation_model = "gemini-2.5-flash"
    settings.gemini_embedding_model = "text-embedding-004"
    settings.embedding_dimensions = 768
    settings.database_url = "postgresql+psycopg://test:test@localhost:5432/test"
    settings.chunk_size = 100
    settings.chunk_overlap = 20
    settings.retrieval_top_k = 5
    settings.max_upload_mb = 25
    settings.log_level = "INFO"
    return settings


@pytest.fixture
def mock_gemini_provider():
    """Mock GeminiProvider for testing."""
    gemini = Mock(spec=GeminiProvider)
    gemini.embed_document.return_value = [0.1] * 768
    gemini.embed_query.return_value = [0.1] * 768
    gemini.answer.return_value = "Test answer based on context"
    return gemini


@pytest.fixture
def mock_repository():
    """Mock CorpusRepository for testing."""
    repo = Mock(spec=CorpusRepository)
    repo.replace_source.return_value = 5
    
    # Create mock CorpusChunk objects for search results
    chunk1 = Mock(spec=CorpusChunk)
    chunk1.source_type = "pdf"
    chunk1.source_id = "doc-001"
    chunk1.title = "Test Document"
    chunk1.chunk_index = 0
    chunk1.page_number = 1
    chunk1.content = "Sample content for testing"
    
    chunk2 = Mock(spec=CorpusChunk)
    chunk2.source_type = "pdf"
    chunk2.source_id = "doc-001"
    chunk2.title = "Test Document"
    chunk2.chunk_index = 1
    chunk2.page_number = 2
    chunk2.content = "More sample content"
    
    repo.search.return_value = [chunk1, chunk2]
    return repo


@pytest.fixture
def sample_text():
    """Sample text for testing chunking."""
    return "This is a test paragraph.\n\nThis is another paragraph."


@pytest.fixture
def long_text():
    """Long text for testing chunking with multiple chunks."""
    return "This is a test. " * 100


@pytest.fixture
def clean_env():
    """Fixture to clean environment variables before and after tests."""
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client for testing."""
    with pytest.importorskip('corpus_ai.providers.gemini'):
        from unittest.mock import patch
        with patch('corpus_ai.providers.gemini.genai.Client') as mock:
            yield mock
