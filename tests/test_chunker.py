import pytest
from unittest.mock import patch, MagicMock

from corpus_ai.ingestion.chunker import semantic_chunking, get_embedding, get_embeddings_for_document

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client"""
    with patch('corpus_ai.ingestion.chunker.get_client') as mock:
        yield mock


def test_semantic_chunking_basic():
    """Test basic semantic chunking"""
    text = "This is a test paragraph.\n\nThis is another paragraph."
    chunks = semantic_chunking(text, chunk_size=50, overlap=10)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_semantic_chunking_empty():
    """Test semantic chunking with empty string"""
    chunks = semantic_chunking("", chunk_size=100, overlap=10)
    assert chunks == []


def test_semantic_chunking_whitespace():
    """Test semantic chunking with whitespace only"""
    chunks = semantic_chunking("   \n\n   ", chunk_size=100, overlap=10)
    assert chunks == []


def test_semantic_chunking_single_paragraph():
    """Test semantic chunking with single paragraph"""
    text = "This is a single paragraph."
    chunks = semantic_chunking(text, chunk_size=100, overlap=10)
    
    assert len(chunks) == 1
    assert chunks[0] == text


def test_semantic_chunking_long_paragraph():
    """Test semantic chunking with long paragraph that needs splitting"""
    text = "This is a very long paragraph that should be split into multiple chunks because it exceeds the chunk size limit."
    chunks = semantic_chunking(text, chunk_size=30, overlap=5)
    
    assert len(chunks) > 1


def test_semantic_chunking_overlap():
    """Test that overlap is applied"""
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = semantic_chunking(text, chunk_size=20, overlap=5)
    
    if len(chunks) > 1:
        # Check that chunks have some overlap
        for i in range(1, len(chunks)):
            # The end of previous chunk should overlap with start of current
            prev_end = chunks[i-1][-5:]
            curr_start = chunks[i][:5]
            # They should share some content
            assert len(set(prev_end) & set(curr_start)) > 0


def test_get_embedding(mock_gemini_client):
    """Test get_embedding function"""
    mock_response = MagicMock()
    mock_response.embedding.values = [0.1] * 768
    mock_gemini_client.return_value.models.embed_content.return_value = mock_response
    
    result = get_embedding("Test text")
    
    assert result == [0.1] * 768
    mock_gemini_client.assert_called_once()


def test_get_embedding_empty(mock_gemini_client):
    """Test get_embedding with empty string"""
    result = get_embedding("")
    
    assert result == []
    mock_gemini_client.assert_not_called()


def test_get_embeddings_for_document(mock_gemini_client):
    """Test get_embeddings_for_document function"""
    mock_response = MagicMock()
    mock_response.embedding.values = [0.1] * 768
    mock_gemini_client.return_value.models.embed_content.return_value = mock_response
    
    text = "This is a test document."
    results = get_embeddings_for_document(text, chunk_size=20, overlap=5)
    
    assert len(results) > 0
    assert all('chunk_id' in r for r in results)
    assert all('chunk' in r for r in results)
    assert all('embedding' in r for r in results)
    assert all('length' in r for r in results)


def test_get_embeddings_for_document_error_handling(mock_gemini_client):
    """Test error handling in get_embeddings_for_document"""
    mock_gemini_client.return_value.models.embed_content.side_effect = Exception("API Error")
    
    text = "Test document."
    results = get_embeddings_for_document(text, chunk_size=20, overlap=5)
    
    # Should return empty list on error
    assert results == []


def test_semantic_chunking_sentence_splitting():
    """Test that long paragraphs are split by sentences"""
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = semantic_chunking(text, chunk_size=25, overlap=5)
    
    assert len(chunks) > 0


def test_semantic_chunking_word_splitting():
    """Test that very long sentences are split by words"""
    text = "This " * 50  # Very long text
    chunks = semantic_chunking(text, chunk_size=30, overlap=5)
    
    assert len(chunks) > 1
