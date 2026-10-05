# ADR-0001: RAG y PostgreSQL con pgvector

- Estado: propuesta inicial
- Contexto: el proyecto debe responder con información de PDFs y de PostgreSQL usando Gemini.
- Decisión: normalizar ambas fuentes a documentos, recuperar fragmentos relevantes y usar PostgreSQL con pgvector como índice vectorial inicial.
- Motivos: mantiene el índice y sus metadatos cerca de PostgreSQL, evita introducir otro servicio antes de conocer el volumen y permite mantener la búsqueda y la fuente SQL con límites de acceso distintos.
- Consecuencias: requiere habilitar `pgvector`; dimensiones y modelo de embeddings deben fijarse antes de crear el esquema. Una migración de modelo puede exigir reindexar.
- Revisión: reevaluar si el volumen, latencia o requisitos operativos superan lo conveniente para un índice vectorial dentro de PostgreSQL.
