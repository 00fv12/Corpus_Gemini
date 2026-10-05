-- Activar la extensión pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Tabla de embeddings normalizados
CREATE TABLE IF NOT EXISTS corpus_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id VARCHAR(255) NOT NULL,
    source_type VARCHAR(50) NOT NULL,    -- 'pdf', 'sql'
    source_name VARCHAR(255) NOT NULL,   -- Nombre del archivo o tabla
    chunk_text TEXT NOT NULL,
    embedding vector(768) NOT NULL,       -- Coincide con text-embedding-004
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIMEZONE DEFAULT CURRENT_TIMESTAMP
);

-- Índice HNSW para búsqueda por distancia cosenoidal (o L2)
CREATE INDEX IF NOT EXISTS idx_corpus_embeddings_vector 
ON corpus_embeddings 
USING hnsw (embedding vector_cosine_ops);