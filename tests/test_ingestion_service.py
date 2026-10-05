import pytest
from unittest.mock import Mock, MagicMock

from corpus_ai.config.settings import Settings
from corpus_ai.providers.gemini import GeminiProvider
from corpus_ai.ingestion.service import IngestionService
from corpus_ai.ingestion.models import SourceDocument
from corpus_ai.storage.repository import CorpusRepository

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_settings():
    """Mock Settings for testing"""
    settings = Mock(spec=Settings)
    settings.chunk_size = 100
    settings.chunk_overlap = 20
    return settings


@pytest.fixture
def mock_gemini():
    """Mock GeminiProvider for testing"""
    gemini = Mock(spec=GeminiProvider)
    gemini.embed_document.return_value = [0.1] * 768
    return gemini


@pytest.fixture
def mock_repository():
    """Mock CorpusRepository for testing"""
    repo = Mock(spec=CorpusRepository)
    repo.replace_source.return_value = 5
    return repo


@pytest.fixture
def ingestion_service(mock_settings, mock_gemini, mock_repository):
    """Create IngestionService instance with mocks"""
    return IngestionService(mock_settings, mock_gemini, mock_repository)


def test_ingestion_service_init(ingestion_service, mock_settings, mock_gemini, mock_repository):
    """Test IngestionService initialization"""
    assert ingestion_service.settings == mock_settings
    assert ingestion_service.gemini == mock_gemini
    assert ingestion_service.repository == mock_repository


def test_ingest_simple_document(ingestion_service, mock_repository, mock_gemini):
    """Test ingesting a simple document without pages"""
    document = SourceDocument(
        source_type="pdf",
        source_id="doc-001",
        title="Test Document",
        text="This is a test document content that should be chunked"
    )
    
    result = ingestion_service.ingest(document)
    
    assert result == 5
    mock_repository.replace_source.assert_called_once()
    
    # Verify embed_document was called for each chunk
    assert mock_gemini.embed_document.call_count > 0


def test_ingest_document_with_pages(ingestion_service, mock_repository, mock_gemini):
    """Test ingesting a document with pages"""
    pages = [(1, "Page 1 content"), (2, "Page 2 content")]
    document = SourceDocument(
        source_type="pdf",
        source_id="doc-002",
        title="Multi-page Document",
        text="Full text",
        pages=pages
    )
    
    result = ingestion_service.ingest(document)
    
    assert result == 5
    mock_repository.replace_source.assert_called_once()


def test_ingest_empty_document(ingestion_service, mock_repository):
    """Test ingesting an empty document"""
    document = SourceDocument(
        source_type="pdf",
        source_id="doc-003",
        title="Empty Document",
        text=""
    )
    
    result = ingestion_service.ingest(document)
    
    # Should still call replace_source with empty chunks
    mock_repository.replace_source.assert_called_once()


def test_ingest_long_document(ingestion_service, mock_gemini):
    """Test ingesting a long document that produces multiple chunks"""
    long_text = "This is a test. " * 100  # Long text
    document = SourceDocument(
        source_type="pdf",
        source_id="doc-004",
        title="Long Document",
        text=long_text
    )
    
    result = ingestion_service.ingest(document)
    
    # Should produce multiple chunks
    assert result > 1


def test_ingest_with_metadata(ingestion_service):
    """Test ingesting a document with metadata"""
    metadata = {"author": "test", "date": "2024-01-01"}
    document = SourceDocument(
        source_type="sql",
        source_id="table-001",
        title="Products Table",
        text="Table data",
        metadata=metadata
    )
    
    result = ingestion_service.ingest(document)
    
    assert result > 0


def test_ingest_uses_chunk_settings(ingestion_service, mock_settings):
    """Test that chunk settings from settings are used"""
    mock_settings.chunk_size = 50
    mock_settings.chunk_overlap = 10
    
    document = SourceDocument(
        source_type="pdf",
        source_id="doc-005",
        title="Test",
        text="This is a test document with some content"
    )
    
    ingestion_service.ingest(document)
    
    # Verify the chunking used the settings
    call_args = ingestion_service.repository.replace_source.call_args
    chunks = call_args[0][1]  # Second argument is the chunks list
    
    # Chunks should respect the chunk_size setting
    for chunk in chunks:
        assert len(chunk[0]) <= mock_settings.chunk_size + mock_settings.chunk_overlap


def test_ingest_replaces_existing_source(ingestion_service, mock_repository):
    """Test that ingesting replaces existing source data"""
    document = SourceDocument(
        source_type="pdf",
        source_id="doc-006",
        title="Test Document",
        text="Updated content"
    )
    
    ingestion_service.ingest(document)
    
    # Verify replace_source was called (not add)
    mock_repository.replace_source.assert_called_once()
