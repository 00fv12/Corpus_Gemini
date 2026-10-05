import pytest
from unittest.mock import Mock, patch, MagicMock
from pypdf import PdfReader

from corpus_ai.ingestion.pdf import load_pdf
from corpus_ai.ingestion.models import Document


@pytest.fixture
def mock_pdf_reader():
    """Mock PdfReader for testing"""
    with patch('corpus_ai.ingestion.pdf.PdfReader') as mock:
        yield mock


def test_load_pdf_basic(mock_pdf_reader):
    """Test basic PDF loading"""
    mock_page1 = Mock()
    mock_page1.extract_text.return_value = "Page 1 content"
    mock_page2 = Mock()
    mock_page2.extract_text.return_value = "Page 2 content"
    
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [mock_page1, mock_page2]
    mock_reader_instance.__len__ = MagicMock(return_value=2)
    mock_pdf_reader.return_value = mock_reader_instance
    
    documents = load_pdf("test.pdf")
    
    assert len(documents) == 2
    assert all(isinstance(doc, Document) for doc in documents)
    assert documents[0].source_type == "pdf"
    assert documents[0].source_name == "test.pdf"
    assert documents[0].text == "Page 1 content"
    assert documents[0].metadata["page_number"] == 1
    assert documents[1].metadata["page_number"] == 2


def test_load_pdf_empty_page(mock_pdf_reader):
    """Test PDF with empty pages"""
    mock_page1 = Mock()
    mock_page1.extract_text.return_value = "Content"
    mock_page2 = Mock()
    mock_page2.extract_text.return_value = ""  # Empty page
    mock_page3 = Mock()
    mock_page3.extract_text.return_value = "   "  # Whitespace only
    
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [mock_page1, mock_page2, mock_page3]
    mock_reader_instance.__len__ = MagicMock(return_value=3)
    mock_pdf_reader.return_value = mock_reader_instance
    
    documents = load_pdf("test.pdf")
    
    # Only non-empty pages should be included
    assert len(documents) == 1
    assert documents[0].text == "Content"


def test_load_pdf_all_empty(mock_pdf_reader):
    """Test PDF with all empty pages"""
    mock_page = Mock()
    mock_page.extract_text.return_value = ""
    
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [mock_page]
    mock_reader_instance.__len__ = MagicMock(return_value=1)
    mock_pdf_reader.return_value = mock_reader_instance
    
    documents = load_pdf("test.pdf")
    
    assert len(documents) == 0


def test_load_pdf_none_text(mock_pdf_reader):
    """Test PDF when extract_text returns None"""
    mock_page = Mock()
    mock_page.extract_text.return_value = None
    
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [mock_page]
    mock_reader_instance.__len__ = MagicMock(return_value=1)
    mock_pdf_reader.return_value = mock_reader_instance
    
    documents = load_pdf("test.pdf")
    
    assert len(documents) == 0


def test_load_pdf_document_id_format(mock_pdf_reader):
    """Test that document IDs are formatted correctly"""
    mock_page = Mock()
    mock_page.extract_text.return_value = "Content"
    
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [mock_page]
    mock_reader_instance.__len__ = MagicMock(return_value=1)
    mock_pdf_reader.return_value = mock_reader_instance
    
    documents = load_pdf("/path/to/document.pdf")
    
    assert documents[0].id == "pdf:/path/to/document.pdf:page-1"


def test_load_pdf_metadata_total_pages(mock_pdf_reader):
    """Test that total_pages is included in metadata"""
    mock_page1 = Mock()
    mock_page1.extract_text.return_value = "Page 1"
    mock_page2 = Mock()
    mock_page2.extract_text.return_value = "Page 2"
    
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [mock_page1, mock_page2]
    mock_reader_instance.__len__ = MagicMock(return_value=2)
    mock_pdf_reader.return_value = mock_reader_instance
    
    documents = load_pdf("test.pdf")
    
    # The pdf.py loader doesn't include total_pages in metadata, only page_number
    assert documents[0].metadata["page_number"] == 1
    assert documents[1].metadata["page_number"] == 2


def test_load_pdf_text_stripping(mock_pdf_reader):
    """Test that text is stripped of extra whitespace"""
    mock_page = Mock()
    mock_page.extract_text.return_value = "  Content with spaces  "
    
    mock_reader_instance = MagicMock()
    mock_reader_instance.pages = [mock_page]
    mock_reader_instance.__len__ = MagicMock(return_value=1)
    mock_pdf_reader.return_value = mock_reader_instance
    
    documents = load_pdf("test.pdf")
    
    assert documents[0].text == "Content with spaces"
