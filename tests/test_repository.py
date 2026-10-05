import pytest
from unittest.mock import Mock, MagicMock

from corpus_ai.storage.repository import CorpusRepository
from corpus_ai.ingestion.models import SourceDocument
from corpus_ai.storage.models import CorpusChunk


@pytest.fixture
def mock_session():
    """Mock SQLAlchemy Session"""
    session = MagicMock()
    session.execute = MagicMock()
    session.scalars = MagicMock()
    session.add_all = MagicMock()
    session.commit = MagicMock()
    return session


@pytest.fixture
def repository(mock_session):
    """Create CorpusRepository instance with mocked session"""
    return CorpusRepository(mock_session)


def test_repository_init(repository, mock_session):
    """Test CorpusRepository initialization"""
    assert repository.session == mock_session


def test_replace_source(repository, mock_session):
    """Test replace_source method"""
    document = SourceDocument(
        source_type="pdf",
        source_id="doc-001",
        title="Test Document",
        text="Content"
    )
    chunks = [
        ("Chunk 1 text", 1, [0.1] * 768),
        ("Chunk 2 text", 2, [0.2] * 768)
    ]
    
    result = repository.replace_source(document, chunks)
    
    assert result == 2
    mock_session.execute.assert_called_once()
    mock_session.add_all.assert_called_once()
    mock_session.commit.assert_called_once()


def test_replace_source_empty_chunks(repository, mock_session):
    """Test replace_source with empty chunks"""
    document = SourceDocument(
        source_type="pdf",
        source_id="doc-001",
        title="Test Document",
        text="Content"
    )
    chunks = []
    
    result = repository.replace_source(document, chunks)
    
    assert result == 0
    mock_session.execute.assert_called_once()
    mock_session.add_all.assert_called_once()
    mock_session.commit.assert_called_once()


def test_search(repository, mock_session):
    """Test search method"""
    embedding = [0.1] * 768
    
    # Create mock CorpusChunk objects
    mock_chunk = Mock(spec=CorpusChunk)
    mock_chunk.source_type = "pdf"
    mock_chunk.source_id = "doc-001"
    mock_chunk.title = "Test"
    mock_chunk.chunk_index = 0
    mock_chunk.page_number = 1
    mock_chunk.content = "Content"
    
    mock_scalars = MagicMock()
    mock_scalars.__iter__ = Mock(return_value=iter([mock_chunk]))
    mock_session.scalars.return_value = mock_scalars
    
    result = repository.search(embedding, top_k=5)
    
    assert len(result) == 1
    assert result[0] == mock_chunk
    mock_session.scalars.assert_called_once()


def test_search_no_results(repository, mock_session):
    """Test search with no results"""
    embedding = [0.1] * 768
    
    mock_scalars = MagicMock()
    mock_scalars.__iter__ = Mock(return_value=iter([]))
    mock_session.scalars.return_value = mock_scalars
    
    result = repository.search(embedding, top_k=5)
    
    assert len(result) == 0


def test_search_default_top_k(repository, mock_session):
    """Test search with default top_k"""
    embedding = [0.1] * 768
    
    mock_scalars = MagicMock()
    mock_scalars.__iter__ = Mock(return_value=iter([]))
    mock_session.scalars.return_value = mock_scalars
    
    repository.search(embedding, top_k=3)
    
    # Verify the statement includes limit
    call_args = mock_session.scalars.call_args
    assert call_args[0][0].limit is not None
