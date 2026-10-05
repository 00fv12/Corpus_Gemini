from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from corpus_ai.ingestion.models import SourceDocument
from corpus_ai.storage.models import CorpusChunk


class CorpusRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def replace_source(self, document: SourceDocument, chunks: list[tuple[str, int | None, list[float]]]) -> int:
        self.session.execute(
            delete(CorpusChunk).where(
                CorpusChunk.source_type == document.source_type,
                CorpusChunk.source_id == document.source_id,
            )
        )
        rows = [
            CorpusChunk(
                source_type=document.source_type,
                source_id=document.source_id,
                title=document.title,
                chunk_index=index,
                page_number=page,
                content=text,
                metadata_json=document.metadata,
                embedding=embedding,
            )
            for index, (text, page, embedding) in enumerate(chunks)
        ]
        self.session.add_all(rows)
        self.session.commit()
        return len(rows)

    def search(self, embedding: list[float], top_k: int) -> list[CorpusChunk]:
        statement = select(CorpusChunk).order_by(CorpusChunk.embedding.cosine_distance(embedding)).limit(top_k)
        return list(self.session.scalars(statement))
