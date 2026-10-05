from corpus_ai.config.settings import Settings
from corpus_ai.ingestion.chunking import split_text
from corpus_ai.ingestion.models import SourceDocument
from corpus_ai.providers.gemini import GeminiProvider
from corpus_ai.storage.repository import CorpusRepository


class IngestionService:
    def __init__(self, settings: Settings, gemini: GeminiProvider, repository: CorpusRepository) -> None:
        self.settings = settings
        self.gemini = gemini
        self.repository = repository

    def ingest(self, document: SourceDocument) -> int:
        prepared: list[tuple[str, int | None, list[float]]] = []
        page_texts = document.pages or [(0, document.text)]
        for page_number, page_text in page_texts:
            text_chunks = split_text(page_text, self.settings.chunk_size, self.settings.chunk_overlap)
            prepared.extend(
                (chunk.text, page_number or None, self.gemini.embed_document(chunk.text))
                for chunk in text_chunks
            )
        return self.repository.replace_source(document, prepared)
