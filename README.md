# Gemini Corpus AI

Proyecto Python para consultar un corpus corporativo usando modelos Gemini. Las fuentes iniciales son documentos PDF y datos seleccionados desde PostgreSQL.

> Estado: esqueleto inicial y decisiones de arquitectura. La lógica de ingesta, indexación y consulta se implementará en una siguiente etapa.

## Objetivo

Construir una aplicación de recuperación aumentada (RAG) que permita responder preguntas con evidencia recuperada de fuentes autorizadas, citando el origen de cada fragmento y evitando completar con datos que no estén en el corpus.

## Fuentes y flujo propuesto

1. **PDF**: lectura del archivo, extracción de texto y metadatos de página.
2. **PostgreSQL de origen**: extracción mediante consultas parametrizadas y una credencial de solo lectura; las filas elegidas se convierten en documentos con identificador, campos permitidos y metadatos.
3. **Normalización**: ambos orígenes producen unidades documentales con `source`, `source_id`, `text`, `metadata` y versión/fecha.
4. **Preparación RAG**: división en fragmentos, generación de embeddings, persistencia de fragmentos y vectores en PostgreSQL con `pgvector`.
5. **Consulta**: recuperación de fragmentos relevantes y generación de respuesta con Gemini, incluyendo referencias a las fuentes.

## Estructura

```text
src/corpus_ai/
├── api/          # Entrada HTTP (se definirá al acordar el contrato)
├── config/       # Configuración y validación de entorno
├── ingestion/    # Adaptadores PDF y PostgreSQL, normalización y chunking
├── providers/    # Cliente Gemini y contratos de modelos
├── retrieval/    # Búsqueda, composición de contexto y respuesta RAG
└── storage/      # Persistencia del índice y repositorios

docs/
├── architecture.md
├── roadmap.md
└── decisions/   # Decisiones de arquitectura (ADR)
tests/           # Pruebas unitarias e integración
```

## Decisiones iniciales

- Python 3.12 o superior.
- SDK oficial `google-genai` para Gemini.
- PostgreSQL como origen SQL y como almacenamiento del índice vectorial mediante `pgvector`; se recomienda separar credenciales/esquemas y usar una conexión de solo lectura para las tablas fuente.
- La extracción desde SQL debe ser explícita y allowlisted. No se aceptarán consultas SQL arbitrarias generadas por el modelo en la primera versión.
- El pipeline de ingesta será independiente del endpoint de preguntas y deberá poder reejecutarse de forma idempotente.
- Las respuestas deberán conservar referencias de origen. Los datos recuperados serán tratados como contenido, no como instrucciones.
- La API HTTP queda pendiente de acordar; FastAPI figura como opción para la primera implementación.

## Configuración local

1. Copiar `.env.example` a `.env`.
2. Completar `GEMINI_API_KEY` y la conexión PostgreSQL cuando comience la implementación.
3. Crear un entorno virtual e instalar el paquete en modo editable:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -e ".[dev]"
```

Las dependencias están declaradas para el proyecto, pero el esqueleto todavía no contiene una aplicación ejecutable.

## Seguridad y privacidad

- No guardar claves, contraseñas ni archivos `.env` en el repositorio.
- Usar variables de entorno o un gestor de secretos en despliegues.
- Conceder a la credencial de lectura únicamente acceso a vistas/tablas necesarias.
- Definir qué información puede enviarse a Gemini antes de procesar datos sensibles o regulados.
- Registrar identificadores de documentos y métricas operativas; evitar registrar contenido sensible por defecto.

## Paso a paso para la primera corrida local

 ## Levantar Podman:

#### Bash

podman-compose up -d

## Generar PDF sintético:

#### Bash

python scripts/generate_sample_pdf.py

## Ejecutar Ingesta:

#### Bash

python -m corpus_ai.cli ingest -f sample_corpus.pdf

# Consultar vía RAG

### Probar consultas RAG:

#### Bash

python -m corpus_ai.cli ask "¿Cuál es el plazo para solicitar reembolsos?"

python -m corpus_ai.cli ask "¿En qué horario atiende el equipo de TI?"

python -m corpus_ai.cli ask "¿Cuál es el costo de las licencias de software?"

-(La tercera pregunta debería responder explícitamente que la información no está disponible en el corpus).

# Consultar vía SQL

### Bash

podman exec -it corpus_postgres psql -U corpus_user -d corpus_db -c "
SELECT 
    id, 
    source_name, 
    document_id, 
    vector_dims(embedding) AS dimensiones, 
    LEFT(chunk_text, 60) AS fragmento_texto, 
    created_at 
FROM corpus_embeddings;
"