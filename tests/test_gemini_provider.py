import pytest
from unittest.mock import Mock, MagicMock, patch
from pydantic import SecretStr

from corpus_ai.config.settings import Settings
from corpus_ai.providers.gemini import GeminiProvider

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_settings():
    """Mock Settings for testing"""
    settings = Mock(spec=Settings)
    settings.gemini_api_key = SecretStr("test_api_key")
    settings.gemini_embedding_model = "text-embedding-004"
    settings.gemini_generation_model = "gemini-2.5-flash"
    settings.embedding_dimensions = 768
    return settings


@pytest.fixture
def gemini_provider(mock_settings):
    """Create GeminiProvider instance with mocked client"""
    with patch('corpus_ai.providers.gemini.genai.Client') as mock_client:
        provider = GeminiProvider(mock_settings)
        provider.client = mock_client.return_value
        yield provider


def test_gemini_provider_init(mock_settings):
    """Test GeminiProvider initialization"""
    with patch('corpus_ai.providers.gemini.genai.Client') as mock_client:
        provider = GeminiProvider(mock_settings)
        
        mock_client.assert_called_once_with(api_key="test_api_key")
        assert provider.settings == mock_settings


def test_embed_document(gemini_provider):
    """Test embed_document method"""
    # Mock the embed_content response
    mock_response = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 768
    mock_response.embeddings = [mock_embedding]
    gemini_provider.client.models.embed_content.return_value = mock_response
    
    result = gemini_provider.embed_document("Test text")
    
    assert result == [0.1] * 768
    gemini_provider.client.models.embed_content.assert_called_once()
    
    # Verify task_type is RETRIEVAL_DOCUMENT
    call_args = gemini_provider.client.models.embed_content.call_args
    assert call_args[1]["config"].task_type == "RETRIEVAL_DOCUMENT"


def test_embed_query(gemini_provider):
    """Test embed_query method"""
    mock_response = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.2] * 768
    mock_response.embeddings = [mock_embedding]
    gemini_provider.client.models.embed_content.return_value = mock_response
    
    result = gemini_provider.embed_query("Query text")
    
    assert result == [0.2] * 768
    call_args = gemini_provider.client.models.embed_content.call_args
    assert call_args[1]["config"].task_type == "RETRIEVAL_QUERY"


def test_embed_no_embeddings(gemini_provider):
    """Test error when no embeddings are returned"""
    mock_response = MagicMock()
    mock_response.embeddings = None
    gemini_provider.client.models.embed_content.return_value = mock_response
    
    with pytest.raises(RuntimeError, match="Gemini no devolvió un embedding"):
        gemini_provider.embed_document("Test text")


def test_embed_none_values(gemini_provider):
    """Test error when embedding values are None"""
    mock_response = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = None
    mock_response.embeddings = [mock_embedding]
    gemini_provider.client.models.embed_content.return_value = mock_response
    
    with pytest.raises(RuntimeError, match="Gemini no devolvió un embedding"):
        gemini_provider.embed_document("Test text")


def test_embed_wrong_dimensions(gemini_provider):
    """Test error when embedding dimensions don't match"""
    mock_response = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1] * 500  # Wrong dimension
    mock_response.embeddings = [mock_embedding]
    gemini_provider.client.models.embed_content.return_value = mock_response
    
    with pytest.raises(RuntimeError, match="La dimensión del embedding no coincide"):
        gemini_provider.embed_document("Test text")


def test_answer(gemini_provider):
    """Test answer method"""
    mock_response = MagicMock()
    mock_response.text = "Generated answer"
    gemini_provider.client.models.generate_content.return_value = mock_response
    
    result = gemini_provider.answer("Question", "Context")
    
    assert result == "Generated answer"
    gemini_provider.client.models.generate_content.assert_called_once()
    
    # Verify the method was called with correct parameters
    call_args = gemini_provider.client.models.generate_content.call_args
    assert "Question" in str(call_args)
    assert "Context" in str(call_args)


def test_answer_empty_response(gemini_provider):
    """Test error when Gemini returns empty response"""
    mock_response = MagicMock()
    mock_response.text = ""
    gemini_provider.client.models.generate_content.return_value = mock_response
    
    with pytest.raises(RuntimeError, match="Gemini devolvió una respuesta vacía"):
        gemini_provider.answer("Question", "Context")


def test_answer_none_response(gemini_provider):
    """Test error when Gemini returns None"""
    mock_response = MagicMock()
    mock_response.text = None
    gemini_provider.client.models.generate_content.return_value = mock_response
    
    with pytest.raises(RuntimeError, match="Gemini devolvió una respuesta vacía"):
        gemini_provider.answer("Question", "Context")


def test_normalization_for_gemini_embedding_001(gemini_provider, mock_settings):
    """Test normalization for gemini-embedding-001 model"""
    mock_settings.gemini_embedding_model = "gemini-embedding-001"
    mock_settings.embedding_dimensions = 3  # Set to match test data
    
    mock_response = MagicMock()
    mock_embedding = MagicMock()
    # Non-normalized values
    mock_embedding.values = [1.0, 2.0, 2.0]  # Norm = 3
    mock_response.embeddings = [mock_embedding]
    gemini_provider.client.models.embed_content.return_value = mock_response
    
    result = gemini_provider.embed_document("Test text")
    
    # After normalization: [1/3, 2/3, 2/3]
    expected_norm = sum(v * v for v in result) ** 0.5
    assert abs(expected_norm - 1.0) < 0.001  # Should be normalized to unit length


def test_no_normalization_for_other_models(gemini_provider, mock_settings):
    """Test that normalization is not applied for other models"""
    mock_settings.gemini_embedding_model = "text-embedding-004"
    mock_settings.embedding_dimensions = 3  # Set to match test data
    
    mock_response = MagicMock()
    mock_embedding = MagicMock()
    original_values = [1.0, 2.0, 3.0]
    mock_embedding.values = original_values[:]
    mock_response.embeddings = [mock_embedding]
    gemini_provider.client.models.embed_content.return_value = mock_response
    
    result = gemini_provider.embed_document("Test text")
    
    # Values should remain unchanged
    assert result == original_values
