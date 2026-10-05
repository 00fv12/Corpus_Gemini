import argparse
import os
import sys
from typing import List
from pypdf import PdfReader
from google import genai

from corpus_ai.ingestion.models import Document
from corpus_ai.storage.vector_store import save_embeddings, search_similar_chunks

# Inicializar cliente oficial de Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("[!] ERROR: La variable de entorno GEMINI_API_KEY no está configurada.", file=sys.stderr)
    sys.exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)


def load_pdf(file_path: str) -> List[Document]:
    """Extrae texto página por página desde un PDF y construye objetos Document."""
    reader = PdfReader(file_path)
    documents = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            doc = Document(
                id=f"pdf:{file_path}:page-{i+1}",
                source_type="pdf",
                source_name=os.path.basename(file_path),
                text=text.strip(),
                metadata={"page_number": i + 1, "total_pages": len(reader.pages)},
            )
            documents.append(doc)

    return documents


def get_embedding(text: str) -> List[float]:
    """Genera embeddings vectoriales de 768 dimensiones con text-embedding-004."""
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text,
    )
    return response.embedding.values


def handle_ingest(args):
    """Pipeline de ingesta: Lee PDF -> Genera Embeddings -> Persiste en pgvector."""
    file_path = args.file
    if not os.path.exists(file_path):
        print(f"[!] Archivo no encontrado: {file_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Leyendo documento: {file_path}")
    documents = load_pdf(file_path)
    print(f"[+] Se extrajeron {len(documents)} páginas/documentos.")

    chunks_to_save = []
    for doc in documents:
        print(f"[*] Procesando embedding para {doc.id}...")
        emb = get_embedding(doc.text)
        chunks_to_save.append({
            "document_id": doc.id,
            "source_type": doc.source_type,
            "source_name": doc.source_name,
            "chunk_text": doc.text,
            "embedding": emb,
            "metadata": doc.metadata,
        })

    print("[*] Insertando registros en PostgreSQL pgvector...")
    save_embeddings(chunks_to_save)
    print(f"[✔] Ingesta completada con éxito. Se indexaron {len(chunks_to_save)} fragmentos.")


def handle_ask(args):
    """Pipeline de consulta: Pregunta -> Embedding -> Búsqueda vectorial -> Síntesis RAG con Gemini."""
    query = args.query
    print(f"[*] Buscando evidencia para la consulta: '{query}'")

    query_emb = get_embedding(query)
    relevant_chunks = search_similar_chunks(query_emb, top_k=args.top_k)

    if not relevant_chunks:
        print("[!] No se encontraron fragmentos relevantes en el corpus.")
        return

    print(f"[+] {len(relevant_chunks)} fragmentos recuperados de la base de datos.")

    # Construcción de contexto con referencias explícitas
    context_blocks = []
    for chunk in relevant_chunks:
        ref = f"Fuente: {chunk['source_name']} (ID: {chunk['document_id']})"
        context_blocks.append(f"[{ref}]\n{chunk['chunk_text']}")

    context_str = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
Responde a la pregunta únicamente utilizando la evidencia proporcionada a continuación.
Si la información no está presente en el contexto, responde explícitamente que no cuentas con esa información.

Contexto recuperado:
{context_str}

Pregunta: {query}

Instrucciones de formato:
- Cita expresamente la fuente de la información citando [Fuente: <nombre_fuente>].
- Mantén la respuesta precisa y fiel al contexto recuperado.
"""

    print("[*] Sintetizando respuesta con Gemini 2.5 Flash...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    print("\n" + "=" * 60)
    print("RESPUESTA:")
    print("=" * 60)
    print(response.text)
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Gemini Corpus AI CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcomando: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Ingresar un archivo PDF al corpus")
    ingest_parser.add_argument("--file", "-f", required=True, help="Ruta al archivo PDF")

    # Subcomando: ask
    ask_parser = subparsers.add_parser("ask", help="Consultar al corpus mediante RAG")
    ask_parser.add_argument("query", type=str, help="Pregunta a realizar")
    ask_parser.add_argument("--top-k", type=int, default=3, help="Cantidad de fragmentos a recuperar")

    args = parser.parse_args()

    if args.command == "ingest":
        handle_ingest(args)
    elif args.command == "ask":
        handle_ask(args)


if __name__ == "__main__":
    main()