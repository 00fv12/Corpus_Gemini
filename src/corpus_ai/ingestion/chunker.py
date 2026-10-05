from google import genai
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def semantic_chunking(text: str,
                      chunk_size: int = 300,
                      overlap: int = 50) -> list[str]:
    """
    Divide texto en chunks respetando límites semánticos
    (párrafos/oraciones) con superposición para mantener contexto.

    Args:
        text: Texto original a dividir
        chunk_size: Tamaño máximo de cada chunk (en caracteres)
        overlap: Número de caracteres de superposición entre chunks

    Returns:
        Lista de strings (chunks)
    """
    if not text or len(text.strip()) == 0:
        return []

    # Dividir por párrafos primero
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # Si agregar este párrafo excede el tamaño
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        else:
            # Guardar chunk actual si existe
            if current_chunk:
                chunks.append(current_chunk)

            # Si el párrafo es muy largo, dividirlo por oraciones
            if len(paragraph) > chunk_size:
                sentences = [s.strip() for s in paragraph.replace('. ', '.|').split('|') if s.strip()]
                for sentence in sentences:
                    if len(sentence) + 2 <= chunk_size:
                        if len(current_chunk) + len(sentence) + 2 <= chunk_size:
                            current_chunk += ". " + sentence if current_chunk and not current_chunk.endswith(
                                '.') else sentence
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence
                    else:
                        # Chunk muy largo, dividir por palabras
                        words = sentence.split()
                        temp_chunk = ""
                        for word in words:
                            if len(temp_chunk) + len(word) + 1 <= chunk_size:
                                temp_chunk += " " + word if temp_chunk else word
                            else:
                                if temp_chunk:
                                    chunks.append(temp_chunk)
                                temp_chunk = word
                        if temp_chunk:
                            current_chunk = temp_chunk
            else:
                current_chunk = paragraph

    # Agregar el último chunk
    if current_chunk:
        chunks.append(current_chunk)

    # Aplicar overlap si hay más de un chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Agregar final del chunk anterior como contexto
                prev_end = chunks[i - 1][-overlap:] if len(chunks[i - 1]) >= overlap else chunks[i - 1]
                chunk = prev_end + " " + chunk
            overlapped_chunks.append(chunk)
        chunks = overlapped_chunks

    return chunks


def get_embedding(text: str) -> list[float]:
    """
    Genera embedding para un texto usando Gemini API.

    Args:
        text: Texto a convertir en embedding

    Returns:
        Lista de floats representando el embedding
    """
    if not text or len(text.strip()) == 0:
        return []

    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return response.embedding.values


def get_embeddings_for_document(text: str,
                                chunk_size: int = 300,
                                overlap: int = 50) -> list[dict]:
    """
    Procesa un documento completo: lo divide en chunks y genera embeddings.

    Args:
        text: Documento completo
        chunk_size: Tamaño de cada chunk
        overlap: Superposición entre chunks

    Returns:
        Lista de diccionarios con 'chunk' y 'embedding'
    """
    chunks = semantic_chunking(text, chunk_size, overlap)

    results = []
    for i, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(chunk)
            results.append({
                'chunk_id': i,
                'chunk': chunk,
                'embedding': embedding,
                'length': len(chunk)
            })
        except Exception as e:
            print(f"Error procesando chunk {i}: {e}")
            continue

    return results


# Ejemplo de uso
if __name__ == "__main__":
    sample_text = """
    La inteligencia artificial está transformando la industria tecnológica.
    Los modelos de lenguaje grandes han revolucionado la forma en que interactuamos con las máquinas.

    El edge computing permite procesar datos cerca de la fuente de generación.
    Esto reduce la latencia y mejora la privacidad de los datos.

    La combinación de IA y edge computing abre nuevas posibilidades para aplicaciones en tiempo real.
    Los dispositivos IoT pueden tomar decisiones inteligentes sin depender completamente de la nube.
    """

    # Procesar documento
    results = get_embeddings_for_document(sample_text, chunk_size=150, overlap=30)

    # Mostrar resultados
    for result in results:
        print(f"\nChunk {result['chunk_id']} ({result['length']} chars):")
        print(f"Texto: {result['chunk'][:100]}...")
        print(f"Embedding dimensión: {len(result['embedding'])}")