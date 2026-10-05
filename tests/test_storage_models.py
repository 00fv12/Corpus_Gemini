import pytest

from corpus_ai.storage.models import CorpusChunk


def test_corpus_chunk_table_name():
    """Test CorpusChunk table name"""
    assert CorpusChunk.__tablename__ == "corpus_chunks"


def test_corpus_chunk_columns():
    """Test CorpusChunk has required columns"""
    # Check that the model has the expected columns
    assert hasattr(CorpusChunk, 'id')
    assert hasattr(CorpusChunk, 'source_type')
    assert hasattr(CorpusChunk, 'source_id')
    assert hasattr(CorpusChunk, 'title')
    assert hasattr(CorpusChunk, 'chunk_index')
    assert hasattr(CorpusChunk, 'page_number')
    assert hasattr(CorpusChunk, 'content')
    assert hasattr(CorpusChunk, 'metadata_json')
    assert hasattr(CorpusChunk, 'embedding')
    assert hasattr(CorpusChunk, 'created_at')


def test_corpus_chunk_unique_constraint():
    """Test CorpusChunk has unique constraint"""
    assert len(CorpusChunk.__table_args__) == 1
    constraint = CorpusChunk.__table_args__[0]
    assert 'source_type' in str(constraint.columns)
    assert 'source_id' in str(constraint.columns)
    assert 'chunk_index' in str(constraint.columns)


def test_corpus_chunk_inherits_from_base():
    """Test CorpusChunk inherits from Base"""
    from corpus_ai.storage.database import Base
    assert issubclass(CorpusChunk, Base)
