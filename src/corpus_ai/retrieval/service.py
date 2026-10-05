from dataclasses import dataclass

from corpus_ai.config.settings import Settings
from corpus_ai.providers.gemini import GeminiProvider
from corpus_ai.storage.repository import CorpusRepository


@dataclass(frozen=True, slots=True)
class Citation:
    citation_id: str
    source_type: str
    source_id: str
    title: str
    chunk_index: int
    page_number: int | None
    excerpt: str


@dataclass(frozen=True, slots=True)
class QueryResult:
    answer: str
    citations: list[Citation]


class RetrievalService:
    def __init__(self, settings: Settings, gemini: GeminiProvider, repository: CorpusRepository) -> None:
        self.settings = settings
        self.gemini = gemini
        self.repository = repository

    def ask(self, question: str) -> QueryResult:
        question = question.strip()
        if not question:
            raise ValueError("La pregunta no puede estar vacía")
        matches = self.repository.search(self.gemini.embed_query(question), self.settings.retrieval_top_k)
        citations = [
            Citation(
                citation_id=f"C{index}",
                source_type=row.source_type,
                source_id=row.source_id,
                title=row.title,
                chunk_index=row.chunk_index,
                page_number=row.page_number,
                excerpt=row.content[:500],
            )
            for index, row in enumerate(matches, start=1)
        ]
        context = "\n\n".join(
            f"[C{index}] {row.title} (fragmento {row.chunk_index + 1}):\n{row.content}"
            for index, row in enumerate(matches, start=1)
        )
        answer = self.gemini.answer(question, context) if matches else "No encontré información en el corpus para responder."
        return QueryResult(answer, citations)
