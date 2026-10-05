import pytest

from corpus_ai.retrieval.service import RetrievalService


def test_retrieval_service_creation():
    """Test RetrievalService can be instantiated"""
    from corpus_ai.config.settings import Settings
    from corpus_ai.providers.gemini import GeminiProvider
    from corpus_ai.storage.repository import CorpusRepository
    from unittest.mock import Mock

    mock_settings = Mock(spec=Settings)
    mock_gemini = Mock(spec=GeminiProvider)
    mock_repo = Mock(spec=CorpusRepository)

    service = RetrievalService(mock_settings, mock_gemini, mock_repo)

    assert service.settings == mock_settings
    assert service.gemini == mock_gemini
    assert service.repository == mock_repo
