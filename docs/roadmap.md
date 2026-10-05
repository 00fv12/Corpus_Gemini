# Hoja de ruta

## Etapa 0 — Base del proyecto
- Estructura del paquete, configuración de ejemplo y documentación inicial.
- Decisiones abiertas registradas sin acoplar la solución a un flujo no acordado.

## Etapa 1 — Primer flujo vertical PDF
- Cargar un PDF de prueba.
- Extraer texto conservando página y metadatos.
- Fragmentar, generar embeddings y persistir en pgvector.
- Consultar con Gemini y devolver citas verificables.

## Etapa 2 — Fuente PostgreSQL
- Seleccionar vistas/tablas autorizadas y definir mapeo a documentos.
- Aplicar lectura incremental e idempotencia.
- Registrar versión/fecha de extracción y estrategia de actualización.

## Etapa 3 — Calidad y operación
- Pruebas de recuperación y respuestas con un conjunto de preguntas esperado.
- Manejo de errores, límites, reintentos y observabilidad sin exponer secretos.
- Autenticación, autorización, retención y despliegue según el entorno elegido.

## Decisiones pendientes
- Entorno de ejecución y forma de despliegue.
- Modelos Gemini concretos para generación y embeddings.
- Tablas/vistas PostgreSQL y sensibilidad de los datos.
- Formato de citas, usuarios y permisos por documento.
- Volumen de PDFs, idiomas, escaneos/OCR y frecuencia de actualización.
