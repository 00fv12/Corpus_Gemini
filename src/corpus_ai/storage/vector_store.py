# src/corpus_ai/storage/vector_store.py
import os
import json
import psycopg
from pgvector.psycopg import register_vector
from typing import List, Dict, Any

DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://corpus_user:corpus_password@localhost:5432/corpus_db"
)

def get_connection():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn

def save_embeddings(chunks: List[Dict[str, Any]]):
    """
    Guarda una lista de fragmentos procesados en la base de datos.
    Cada dict debe contener: document_id, source_type, source_name, chunk_text, embedding, metadata
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            for chunk in chunks:
                cur.execute(
                    """
                    INSERT INTO corpus_embeddings 
                    (document_id, source_type, source_name, chunk_text, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        chunk["document_id"],
                        chunk["source_type"],
                        chunk["source_name"],
                        chunk["chunk_text"],
                        chunk["embedding"],
                        json.dumps(chunk.get("metadata", {}))
                    )
                )
        conn.commit()

def search_similar_chunks(query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Recupera los fragmentos más similares utilizando la distancia coseno (<=>).
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT document_id, source_type, source_name, chunk_text, metadata, 
                       1 - (embedding <=> %s::vector) AS similarity
                FROM corpus_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_embedding, query_embedding, top_k)
            )
            rows = cur.fetchall()
            
            results = []
            for row in rows:
                results.append({
                    "document_id": row[0],
                    "source_type": row[1],
                    "source_name": row[2],
                    "chunk_text": row[3],
                    "metadata": row[4],
                    "similarity": row[5]
                })
            return results