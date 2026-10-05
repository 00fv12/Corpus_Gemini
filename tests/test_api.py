import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock

from corpus_ai.api.main import app, get_settings_dependency


@pytest.fixture
def mock_settings():
    """Mock Settings for testing"""
    settings = Mock()
    settings.gemini_api_key = Mock()
    settings.gemini_api_key.get_secret_value.return_value = "test_key"
    settings.gemini_generation_model = "gemini-2.5-flash"
    settings.gemini_embedding_model = "text-embedding-004"
    settings.embedding_dimensions = 768
    settings.retrieval_top_k = 5
    settings.chunk_size = 1400
    settings.chunk_overlap = 200
    return settings


@pytest.fixture
def client(mock_settings):
    """Create a test client for the FastAPI app with mocked settings"""
    app.dependency_overrides[get_settings_dependency] = lambda: mock_settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Gemini Corpus AI API"
    assert data["version"] == "0.1.0"
    assert data["docs"] == "/docs"
    assert data["redoc"] == "/redoc"


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@patch('corpus_ai.api.main.get_session')
@patch('corpus_ai.api.main.GeminiProvider')
@patch('corpus_ai.api.main.IngestionService')
def test_ingest_document(mock_ingestion_service, mock_gemini_provider, mock_get_session, client):
    """Test document ingestion endpoint"""
    # Mock the session context manager
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session
    
    # Mock the ingestion service
    mock_service_instance = MagicMock()
    mock_service_instance.ingest.return_value = 5
    mock_ingestion_service.return_value = mock_service_instance
    
    request_data = {
        "source_type": "pdf",
        "source_id": "doc-001",
        "title": "Test Document",
        "text": "This is a test document content"
    }
    
    response = client.post("/ingest", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Documento ingerido exitosamente"
    assert data["chunks_count"] == 5
    assert data["source_id"] == "doc-001"


@patch('corpus_ai.api.main.get_session')
@patch('corpus_ai.api.main.GeminiProvider')
@patch('corpus_ai.api.main.RetrievalService')
def test_query_corpus(mock_retrieval_service, mock_gemini_provider, mock_get_session, client):
    """Test query endpoint"""
    # Mock the session context manager
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session
    
    # Mock the retrieval service
    mock_service_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.answer = "Test answer"
    mock_result.citations = []
    mock_service_instance.ask.return_value = mock_result
    mock_retrieval_service.return_value = mock_service_instance
    
    request_data = {
        "question": "What is this about?",
        "top_k": 3
    }
    
    response = client.post("/query", json=request_data)
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test answer"
    assert data["citations"] == []


@patch('corpus_ai.api.main.get_session')
@patch('corpus_ai.api.main.GeminiProvider')
@patch('corpus_ai.api.main.RetrievalService')
def test_query_corpus_default_top_k(mock_retrieval_service, mock_gemini_provider, mock_get_session, client):
    """Test query endpoint with default top_k"""
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session
    
    mock_service_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.answer = "Test answer"
    mock_result.citations = []
    mock_service_instance.ask.return_value = mock_result
    mock_retrieval_service.return_value = mock_service_instance
    
    request_data = {
        "question": "Test question"
    }
    
    response = client.post("/query", json=request_data)
    
    assert response.status_code == 200


@patch('corpus_ai.api.main.get_session')
@patch('corpus_ai.api.main.GeminiProvider')
@patch('corpus_ai.api.main.RetrievalService')
def test_query_corpus_empty_question(mock_retrieval_service, mock_gemini_provider, mock_get_session, client):
    """Test query endpoint with empty question (should fail at service level)"""
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session
    
    mock_service_instance = MagicMock()
    mock_service_instance.ask.side_effect = ValueError("La pregunta no puede estar vacía")
    mock_retrieval_service.return_value = mock_service_instance
    
    request_data = {
        "question": ""
    }
    
    response = client.post("/query", json=request_data)
    
    assert response.status_code == 400


@patch('corpus_ai.api.main.get_session')
@patch('corpus_ai.api.main.GeminiProvider')
@patch('corpus_ai.api.main.IngestionService')
def test_ingest_pdf(mock_ingestion_service, mock_gemini_provider, mock_get_session, client):
    """Test PDF ingestion endpoint"""
    mock_session = MagicMock()
    mock_get_session.return_value.__enter__.return_value = mock_session
    
    mock_service_instance = MagicMock()
    mock_service_instance.ingest.return_value = 3
    mock_ingestion_service.return_value = mock_service_instance
    
    # Create a mock PDF file
    pdf_content = b"%PDF-1.4\n%test pdf content"
    files = {"file": ("test.pdf", pdf_content, "application/pdf")}
    
    with patch('pypdf.PdfReader') as mock_pdf_reader:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Sample PDF text content"
        mock_reader_instance = MagicMock()
        mock_reader_instance.pages = [mock_page]
        mock_reader_instance.__len__ = MagicMock(return_value=1)
        mock_pdf_reader.return_value = mock_reader_instance
        
        response = client.post("/ingest/pdf", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "PDF ingerido exitosamente"
    assert data["chunks_count"] == 3
    assert data["filename"] == "test.pdf"
    assert data["pages"] == 1


