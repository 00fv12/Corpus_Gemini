# src/corpus_ai/retrieval/models.py
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Citation:
    source_type: str
    source_name: str
    document_id: str
    snippet: str

@dataclass
class RAGResponse:
    answer: str
    citations: List[Citation]