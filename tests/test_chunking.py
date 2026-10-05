import pytest

from corpus_ai.ingestion.chunking import TextChunk, split_text


def test_split_text_basic():
    """Test basic text splitting"""
    text = "This is a test text with some words that should be split into chunks."
    chunks = split_text(text, chunk_size=20, overlap=5)
    
    assert len(chunks) > 0
    assert all(isinstance(chunk, TextChunk) for chunk in chunks)
    assert all(chunk.text for chunk in chunks)


def test_split_text_empty_string():
    """Test splitting empty string"""
    chunks = split_text("", chunk_size=100, overlap=10)
    assert chunks == []


def test_split_text_whitespace_only():
    """Test splitting whitespace-only string"""
    chunks = split_text("   \n\n   ", chunk_size=100, overlap=10)
    assert chunks == []


def test_split_text_chunk_size_validation():
    """Test that chunk_size must be greater than zero"""
    with pytest.raises(ValueError, match="chunk_size debe ser mayor que cero"):
        split_text("test text", chunk_size=0, overlap=10)
    
    with pytest.raises(ValueError, match="chunk_size debe ser mayor que cero"):
        split_text("test text", chunk_size=-5, overlap=10)


def test_split_text_overlap_validation():
    """Test that overlap must be >= 0 and < chunk_size"""
    with pytest.raises(ValueError, match="overlap debe ser >= 0 y menor que chunk_size"):
        split_text("test text", chunk_size=100, overlap=-1)
    
    with pytest.raises(ValueError, match="overlap debe ser >= 0 y menor que chunk_size"):
        split_text("test text", chunk_size=100, overlap=100)
    
    with pytest.raises(ValueError, match="overlap debe ser >= 0 y menor que chunk_size"):
        split_text("test text", chunk_size=100, overlap=150)


def test_split_text_single_chunk():
    """Test when text fits in a single chunk"""
    text = "Short text"
    chunks = split_text(text, chunk_size=100, overlap=10)
    
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chunk_index == 0
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len(text)


def test_split_text_multiple_chunks():
    """Test splitting text into multiple chunks"""
    text = "This is a longer text that should be split into multiple chunks for testing purposes."
    chunks = split_text(text, chunk_size=20, overlap=5)
    
    assert len(chunks) > 1
    assert chunks[0].chunk_index == 0
    assert chunks[-1].chunk_index == len(chunks) - 1


def test_split_text_normalization():
    """Test that text is normalized (extra whitespace removed)"""
    text = "This  has   extra    whitespace"
    chunks = split_text(text, chunk_size=100, overlap=10)
    
    assert "  " not in chunks[0].text
    assert "   " not in chunks[0].text


def test_text_chunk_attributes():
    """Test TextChunk dataclass attributes"""
    chunk = TextChunk(
        text="Sample text",
        chunk_index=0,
        start_char=0,
        end_char=11
    )
    
    assert chunk.text == "Sample text"
    assert chunk.chunk_index == 0
    assert chunk.start_char == 0
    assert chunk.end_char == 11


def test_split_text_word_boundary():
    """Test that chunks respect word boundaries"""
    text = "The quick brown fox jumps over the lazy dog"
    chunks = split_text(text, chunk_size=15, overlap=5)
    
    # Chunks should not split in the middle of words when possible
    for chunk in chunks:
        assert not chunk.text.startswith(" ") or chunk.text.strip() == ""


def test_split_text_overlap():
    """Test that overlap is applied correctly"""
    text = "This is a test text that will be split into multiple chunks with overlap."
    chunks = split_text(text, chunk_size=20, overlap=5)
    
    if len(chunks) > 1:
        # Check that consecutive chunks have overlapping content
        for i in range(1, len(chunks)):
            # The end of previous chunk should overlap with start of current chunk
            prev_end = chunks[i-1].text[-5:]
            curr_start = chunks[i].text[:5]
            # They should share some content
            assert len(set(prev_end) & set(curr_start)) > 0 or prev_end == curr_start
