import pytest
from datetime import datetime

from corpus_ai.ingestion.models import Document, SourceDocument


def test_document_creation():
    """Test Document dataclass creation"""
    doc = Document(
        id="pdf:doc-01:page-1",
        source_type="pdf",
        source_name="manual.pdf",
        text="Sample text"
    )
    
    assert doc.id == "pdf:doc-01:page-1"
    assert doc.source_type == "pdf"
    assert doc.source_name == "manual.pdf"
    assert doc.text == "Sample text"
    assert doc.metadata == {}
    assert isinstance(doc.created_at, datetime)


def test_document_with_metadata():
    """Test Document with custom metadata"""
    metadata = {"page": 1, "author": "test"}
    doc = Document(
        id="sql:products:101",
        source_type="sql",
        source_name="products",
        text="Product data",
        metadata=metadata
    )
    
    assert doc.metadata == metadata
    assert doc.metadata["page"] == 1
    assert doc.metadata["author"] == "test"


def test_source_document_creation():
    """Test SourceDocument dataclass creation"""
    doc = SourceDocument(
        source_type="pdf",
        source_id="doc-001",
        title="Test Document",
        text="Document content"
    )
    
    assert doc.source_type == "pdf"
    assert doc.source_id == "doc-001"
    assert doc.title == "Test Document"
    assert doc.text == "Document content"
    assert doc.metadata == {}
    assert doc.pages is None


def test_source_document_with_pages():
    """Test SourceDocument with pages"""
    pages = [(1, "Page 1 content"), (2, "Page 2 content")]
    doc = SourceDocument(
        source_type="pdf",
        source_id="doc-002",
        title="Multi-page Document",
        text="Full text",
        pages=pages
    )
    
    assert doc.pages == pages
    assert len(doc.pages) == 2
    assert doc.pages[0] == (1, "Page 1 content")


def test_source_document_with_metadata():
    """Test SourceDocument with metadata"""
    metadata = {"author": "test", "date": "2024-01-01"}
    doc = SourceDocument(
        source_type="sql",
        source_id="table-001",
        title="Products Table",
        text="Table data",
        metadata=metadata
    )
    
    assert doc.metadata == metadata
    assert doc.metadata["author"] == "test"
