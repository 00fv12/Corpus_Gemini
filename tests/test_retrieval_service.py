import pytest
from unittest.mock import Mock, MagicMock

from corpus_ai.config.settings import Settings
from corpus_ai.providers.gemini import GeminiProvider
from corpus_ai.retrieval.service import RetrievalService, Citation, QueryResult
from corpus_ai.storage.models import CorpusChunk


@pytest.fixture
def mock_settings():
    """Mock Settings for testing"""
    settings = Mock(spec=Settings)
    settings.retrieval_top_k = 5
    return settings


@pytest.fixture
def mock_gemini():
    """Mock GeminiProvider for testing"""
    gemini = Mock(spec=GeminiProvider)
    gemini.embed_query.return_value = [0.1] * 768
    gemini.answer.return_value = "Test answer based on context"
    return gemini


@pytest.fixture
def mock_repository():
    """Mock CorpusRepository for testing"""
    repo = Mock()
    
    # Create mock CorpusChunk objects
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
def retrieval_service(mock_settings, mock_gemini, mock_repository):
    """Create RetrievalService instance with mocks"""
    return RetrievalService(mock_settings, mock_gemini, mock_repository)


def test_retrieval_service_init(retrieval_service, mock_settings, mock_gemini, mock_repository):
    """Test RetrievalService initialization"""
    assert retrieval_service.settings == mock_settings
    assert retrieval_service.gemini == mock_gemini
    assert retrieval_service.repository == mock_repository


def test_ask_with_results(retrieval_service, mock_repository, mock_gemini):
    """Test ask method when results are found"""
    result = retrieval_service.ask("What is the test about?")
    
    assert isinstance(result, QueryResult)
    assert result.answer == "Test answer based on context"
    assert len(result.citations) == 2
    
    # Verify repository.search was called
    mock_repository.search.assert_called_once()
    
    # Verify gemini.answer was called
    mock_gemini.answer.assert_called_once()


def test_ask_empty_question(retrieval_service):
    """Test ask method with empty question"""
    with pytest.raises(ValueError, match="La pregunta no puede estar vacía"):
        retrieval_service.ask("")
    
    with pytest.raises(ValueError, match="La pregunta no puede estar vacía"):
        retrieval_service.ask("   ")


def test_ask_no_results(retrieval_service, mock_repository, mock_gemini):
    """Test ask method when no results are found"""
    mock_repository.search.return_value = []
    
    result = retrieval_service.ask("Unknown topic")
    
    assert result.answer == "No encontré información en el corpus para responder."
    assert len(result.citations) == 0
    
    # Verify gemini.answer was NOT called when no results
    mock_gemini.answer.assert_not_called()


def test_citation_creation():
    """Test Citation dataclass"""
    citation = Citation(
        citation_id="C1",
        source_type="pdf",
        source_id="doc-001",
        title="Test Document",
        chunk_index=0,
        page_number=1,
        excerpt="Sample excerpt"
    )
    
    assert citation.citation_id == "C1"
    assert citation.source_type == "pdf"
    assert citation.source_id == "doc-001"
    assert citation.title == "Test Document"
    assert citation.chunk_index == 0
    assert citation.page_number == 1
    assert citation.excerpt == "Sample excerpt"


def test_query_result_creation():
    """Test QueryResult dataclass"""
    citations = [
        Citation(
            citation_id="C1",
            source_type="pdf",
            source_id="doc-001",
            title="Test Document",
            chunk_index=0,
            page_number=1,
            excerpt="Sample excerpt"
        )
    ]
    
    result = QueryResult(answer="Test answer", citations=citations)
    
    assert result.answer == "Test answer"
    assert len(result.citations) == 1
    assert result.citations[0].citation_id == "C1"


def test_ask_question_stripping(retrieval_service, mock_repository):
    """Test that question is stripped before processing"""
    retrieval_service.ask("  Test question  ")
    
    # Verify the question was stripped when calling embed_query
    call_args = retrieval_service.gemini.embed_query.call_args
    assert call_args[0][0] == "Test question"


def test_citation_excerpt_truncation(retrieval_service, mock_repository):
    """Test that citation excerpts are truncated to 500 characters"""
    # Create a chunk with long content
    long_chunk = Mock(spec=CorpusChunk)
    long_chunk.source_type = "pdf"
    long_chunk.source_id = "doc-001"
    long_chunk.title = "Test Document"
    long_chunk.chunk_index = 0
    long_chunk.page_number = 1
    long_chunk.content = "A" * 1000  # 1000 characters
    
    mock_repository.search.return_value = [long_chunk]
    
    result = retrieval_service.ask("Test question")
    
    assert len(result.citations[0].excerpt) <= 500
