from google import genai
import os

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PROMPT_TEMPLATE = """
Responde a la pregunta únicamente utilizando la evidencia proporcionada a continuación.
Si la información no está presente en el contexto, indica explícitamente que no se cuenta con esa información.

Contexto recuperado:
{context}

Pregunta: {question}

Instrucciones adicionales:
- Incluye referencias explicitas al origen de los datos en tu respuesta según los metadatos proporcionados en el contexto.
"""

def generate_rag_response(question: str, retrieved_chunks: list[dict]) -> str:
    context_str = "\n---\n".join([
        f"Origen: {item['source']} (ID: {item['id']})\nContenido: {item['text']}"
        for item in retrieved_chunks
    ])
    
    prompt = PROMPT_TEMPLATE.format(context=context_str, question=question)
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text