# Arquitectura inicial

## Componentes

- **Adaptadores de fuente**: PDF y PostgreSQL. Cada uno traduce su formato a un documento común.
- **Pipeline de ingesta**: valida, normaliza, segmenta, crea embeddings y actualiza el índice.
- **Índice**: PostgreSQL + pgvector almacena fragmentos, embeddings y metadatos de procedencia.
- **Recuperador**: busca fragmentos por similitud y aplica filtros de metadatos/autorización.
- **Generador**: Gemini recibe la pregunta y el contexto recuperado; debe responder con citas y declarar cuando no haya evidencia suficiente.
- **API**: interfaz HTTP pendiente de definir, con operaciones separadas para ingesta y consulta.

## Separación recomendada en PostgreSQL

- Base/esquema de origen: tablas de negocio o vistas autorizadas; acceso de solo lectura.
- Esquema de índice: documentos normalizados, fragmentos y embeddings; escritura desde el proceso de ingesta.
- No mezclar automáticamente la estructura del origen con el modelo interno del índice.

## Flujo de consulta

```text
Pregunta -> validación -> embedding de consulta -> recuperación top-k
         -> contexto con fuentes -> Gemini -> respuesta + referencias
```

## Contrato documental tentativo

```text
Document:
  id: identificador estable
  source: pdf | postgres
  source_id: ruta/archivo o clave primaria lógica
  text: contenido extraído
  metadata: campos permitidos (página, tabla, fecha, etiquetas)
  content_hash: huella para detectar cambios
```

El esquema definitivo y las reglas de chunking se decidirán al implementar la primera fuente.
