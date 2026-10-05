from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextChunk:
    text: str
    chunk_index: int
    start_char: int
    end_char: int


def split_text(text: str, chunk_size: int = 1400, overlap: int = 200) -> list[TextChunk]:
    if chunk_size < 1:
        raise ValueError("chunk_size debe ser mayor que cero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap debe ser >= 0 y menor que chunk_size")
    normalized = " ".join(text.split())
    if not normalized:
        return []
    chunks: list[TextChunk] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        body = normalized[start:end].strip()
        if body:
            chunks.append(TextChunk(body, len(chunks), start, end))
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)
    return chunks
