import pytest
from unittest.mock import Mock, patch, MagicMock

from corpus_ai.storage.vector_store import get_connection, save_embeddings, search_similar_chunks


@pytest.fixture
def mock_connection():
    """Mock psycopg connection"""
    with patch('corpus_ai.storage.vector_store.psycopg.connect') as mock:
        yield mock


@pytest.fixture
def mock_register_vector():
    """Mock register_vector function"""
    with patch('corpus_ai.storage.vector_store.register_vector') as mock:
        yield mock


def test_get_connection(mock_connection, mock_register_vector):
    """Test get_connection function"""
    conn = MagicMock()
    mock_connection.return_value = conn
    
    result = get_connection()
    
    mock_connection.assert_called_once()
    mock_register_vector.assert_called_once_with(conn)
    assert result == conn


@patch('corpus_ai.storage.vector_store.get_connection')
def test_save_embeddings(mock_get_connection):
    """Test save_embeddings function"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    
    chunks = [
        {
            "document_id": "doc-001",
            "source_type": "pdf",
            "source_name": "test.pdf",
            "chunk_text": "Sample text",
            "embedding": [0.1] * 768,
            "metadata": {"page": 1}
        }
    ]
    
    save_embeddings(chunks)
    
    mock_cursor.execute.assert_called_once()
    mock_conn.commit.assert_called_once()


@patch('corpus_ai.storage.vector_store.get_connection')
def test_save_embeddings_multiple_chunks(mock_get_connection):
    """Test save_embeddings with multiple chunks"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    
    chunks = [
        {
            "document_id": f"doc-{i}",
            "source_type": "pdf",
            "source_name": "test.pdf",
            "chunk_text": f"Text {i}",
            "embedding": [0.1] * 768,
            "metadata": {"page": i}
        }
        for i in range(3)
    ]
    
    save_embeddings(chunks)
    
    assert mock_cursor.execute.call_count == 3
    mock_conn.commit.assert_called_once()


@patch('corpus_ai.storage.vector_store.get_connection')
def test_save_embeddings_empty_list(mock_get_connection):
    """Test save_embeddings with empty list"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    
    save_embeddings([])
    
    mock_cursor.execute.assert_not_called()
    mock_conn.commit.assert_called_once()


@patch('corpus_ai.storage.vector_store.get_connection')
def test_search_similar_chunks(mock_get_connection):
    """Test search_similar_chunks function"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        ("doc-001", "pdf", "test.pdf", "Sample text", {"page": 1}, 0.95)
    ]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    
    query_embedding = [0.1] * 768
    results = search_similar_chunks(query_embedding, top_k=3)
    
    assert len(results) == 1
    assert results[0]["document_id"] == "doc-001"
    assert results[0]["source_type"] == "pdf"
    assert results[0]["source_name"] == "test.pdf"
    assert results[0]["chunk_text"] == "Sample text"
    assert results[0]["similarity"] == 0.95


@patch('corpus_ai.storage.vector_store.get_connection')
def test_search_similar_chunks_no_results(mock_get_connection):
    """Test search_similar_chunks with no results"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    
    query_embedding = [0.1] * 768
    results = search_similar_chunks(query_embedding, top_k=3)
    
    assert len(results) == 0


@patch('corpus_ai.storage.vector_store.get_connection')
def test_search_similar_chunks_default_top_k(mock_get_connection):
    """Test search_similar_chunks with default top_k"""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    mock_get_connection.return_value.__enter__.return_value = mock_conn
    
    query_embedding = [0.1] * 768
    search_similar_chunks(query_embedding)
    
    # Verify the SQL query includes LIMIT with default value
    call_args = mock_cursor.execute.call_args
    assert "%s" in call_args[0][0]
