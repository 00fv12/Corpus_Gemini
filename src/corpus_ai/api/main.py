from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os

from corpus_ai.config.settings import get_settings, Settings
from corpus_ai.ingestion.service import IngestionService
from corpus_ai.ingestion.models import SourceDocument
from corpus_ai.retrieval.service import RetrievalService
from corpus_ai.providers.gemini import GeminiProvider
from corpus_ai.storage.database import get_session
from corpus_ai.storage.repository import CorpusRepository

app = FastAPI(
    title="Gemini Corpus AI API",
    description="API para consulta de corpus corporativo usando modelos Gemini con RAG",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


def get_settings_dependency() -> Settings:
    return get_settings()


class IngestRequest(BaseModel):
    source_type: str
    source_id: str
    title: str
    text: str
    metadata: Optional[dict] = None


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict]


@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "Gemini Corpus AI API",
        "version": "0.1.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
async def health_check():
    """Endpoint de verificación de salud"""
    return {"status": "healthy"}


@app.post("/ingest", response_model=dict)
async def ingest_document(request: IngestRequest, settings: Settings = Depends(get_settings_dependency)):
    """
    Ingesta un documento en el corpus
    
    - **source_type**: Tipo de fuente (pdf, sql, etc.)
    - **source_id**: Identificador único de la fuente
    - **title**: Título del documento
    - **text**: Contenido del documento
    - **metadata**: Metadatos adicionales (opcional)
    """
    try:
        with get_session() as session:
            repository = CorpusRepository(session)
            gemini = GeminiProvider(settings)
            ingestion_service = IngestionService(settings, gemini, repository)
            
            document = SourceDocument(
                source_type=request.source_type,
                source_id=request.source_id,
                title=request.title,
                text=request.text,
                metadata=request.metadata or {},
                pages=None
            )
            
            chunks_count = ingestion_service.ingest(document)
            
            return {
                "message": "Documento ingerido exitosamente",
                "chunks_count": chunks_count,
                "source_id": request.source_id
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/pdf", response_model=dict)
async def ingest_pdf(file: UploadFile = File(...), settings: Settings = Depends(get_settings_dependency)):
    """
    Ingesta un archivo PDF en el corpus
    
    - **file**: Archivo PDF a procesar
    """
    try:
        import tempfile
        from pypdf import PdfReader
        
        # Guardar archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Extraer texto del PDF
            reader = PdfReader(tmp_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            
            if not text.strip():
                raise HTTPException(status_code=400, detail="El PDF no contiene texto extraíble")
            
            # Ingestar el documento
            with get_session() as session:
                repository = CorpusRepository(session)
                gemini = GeminiProvider(settings)
                ingestion_service = IngestionService(settings, gemini, repository)
                
                document = SourceDocument(
                    source_type="pdf",
                    source_id=file.filename,
                    title=file.filename,
                    text=text,
                    metadata={"pages": len(reader.pages)},
                    pages=None
                )
                
                chunks_count = ingestion_service.ingest(document)
                
                return {
                    "message": "PDF ingerido exitosamente",
                    "chunks_count": chunks_count,
                    "filename": file.filename,
                    "pages": len(reader.pages)
                }
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_corpus(request: QueryRequest, settings: Settings = Depends(get_settings_dependency)):
    """
    Consulta el corpus usando RAG
    
    - **question**: Pregunta a realizar
    - **top_k**: Cantidad de fragmentos a recuperar (default: 5)
    """
    try:
        with get_session() as session:
            repository = CorpusRepository(session)
            gemini = GeminiProvider(settings)
            retrieval_service = RetrievalService(settings, gemini, repository)
            
            result = retrieval_service.ask(request.question)
            
            return QueryResponse(
                answer=result.answer,
                citations=[
                    {
                        "citation_id": c.citation_id,
                        "source_type": c.source_type,
                        "source_id": c.source_id,
                        "title": c.title,
                        "chunk_index": c.chunk_index,
                        "page_number": c.page_number,
                        "excerpt": c.excerpt
                    }
                    for c in result.citations
                ]
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
