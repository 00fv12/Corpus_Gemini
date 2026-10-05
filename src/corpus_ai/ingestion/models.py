# src/corpus_ai/ingestion/models.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple

@dataclass
class Document:
    id: str  # Ej: "pdf:doc-01:page-1" o "sql:productos:101"
    source_type: str  # "pdf" | "sql"
    source_name: str  # Ej: "manual_operaciones.pdf" o "tabla_productos"
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SourceDocument:
    source_type: str
    source_id: str
    title: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    pages: Optional[List[Tuple[int, str]]] = None  # (page_number, text)