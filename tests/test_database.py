import pytest
from unittest.mock import patch, MagicMock

from corpus_ai.storage.database import get_engine, get_session_factory, get_session, Base


def test_get_engine():
    """Test get_engine function"""
    with patch('corpus_ai.storage.database.create_engine') as mock_create_engine:
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        engine1 = get_engine()
        engine2 = get_engine()
        
        # Should return the same cached instance
        assert engine1 is engine2
        mock_create_engine.assert_called_once()


def test_get_session_factory():
    """Test get_session_factory function"""
    with patch('corpus_ai.storage.database.get_engine') as mock_get_engine:
        mock_engine = MagicMock()
        mock_get_engine.return_value = mock_engine
        
        with patch('corpus_ai.storage.database.sessionmaker') as mock_sessionmaker:
            mock_factory = MagicMock()
            mock_sessionmaker.return_value = mock_factory
            
            factory = get_session_factory()
            
            assert factory == mock_factory
            mock_sessionmaker.assert_called_once()


def test_get_session_context_manager():
    """Test get_session context manager"""
    with patch('corpus_ai.storage.database.get_session_factory') as mock_get_factory:
        mock_factory = MagicMock()
        mock_session = MagicMock()
        mock_factory.return_value = mock_session
        mock_get_factory.return_value = mock_factory
        
        with get_session() as session:
            assert session == mock_session
        
        # Verify commit was called
        mock_session.commit.assert_called_once()
        # Verify close was called
        mock_session.close.assert_called_once()


def test_get_session_rollback_on_error():
    """Test get_session rolls back on error"""
    with patch('corpus_ai.storage.database.get_session_factory') as mock_get_factory:
        mock_factory = MagicMock()
        mock_session = MagicMock()
        mock_factory.return_value = mock_session
        mock_get_factory.return_value = mock_factory
        
        with pytest.raises(ValueError):
            with get_session() as session:
                raise ValueError("Test error")
        
        # Verify rollback was called
        mock_session.rollback.assert_called_once()
        # Verify close was still called
        mock_session.close.assert_called_once()


def test_base_class():
    """Test Base class is DeclarativeBase"""
    from sqlalchemy.orm import DeclarativeBase
    assert isinstance(Base, type)
    assert issubclass(Base, DeclarativeBase)
