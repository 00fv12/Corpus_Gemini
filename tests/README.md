# Pruebas

Este directorio contiene las pruebas unitarias y de integración para el proyecto Corpus AI con Gemini, PostgreSQL y pgvector.

## Estructura de Pruebas

### Pruebas Unitarias
Las pruebas unitarias prueban componentes individuales con mocks, sin dependencias externas:

- **test_config.py**: Configuración y settings de la aplicación
- **test_chunker.py**: Chunking semántico de documentos
- **test_chunking.py**: Funciones adicionales de chunking
- **test_gemini_provider.py**: Proveedor de embeddings y generación de Gemini
- **test_pdf_loader.py**: Carga y extracción de texto desde PDFs
- **test_ingestion_models.py**: Modelos de datos para ingestión
- **test_ingestion_service.py**: Servicio de ingestión de documentos
- **test_retrieval_models.py**: Modelos de datos para recuperación
- **test_retrieval_service.py**: Servicio de recuperación RAG
- **test_repository.py**: Repositorio de datos
- **test_sql_mapper.py**: Mapeo SQL
- **test_storage_models.py**: Modelos de almacenamiento
- **test_database.py**: Conexión y sesión de base de datos
- **test_vector_store.py**: Almacenamiento de vectores con pgvector
- **test_api.py**: Endpoints de la API FastAPI

### Pruebas de Integración
Las pruebas de integración son opt-in y usan datos sintéticos. No dependen de credenciales reales por defecto.

Para ejecutar pruebas de integración:
```bash
pytest -m integration
```

## Ejecutar Pruebas

### Ejecutar todas las pruebas unitarias
```bash
pytest
```

### Ejecutar con cobertura
```bash
pytest --cov=corpus_ai --cov-report=html --cov-report=term
```

### Ejecutar un archivo específico
```bash
pytest tests/test_config.py
```

### Ejecutar una prueba específica
```bash
pytest tests/test_config.py::test_settings_default_values
```

### Ejecutar con verbosidad
```bash
pytest -v
```

### Ejecutar solo pruebas que fallen en la última ejecución
```bash
pytest --lf
```

## Configuración

Las pruebas usan la configuración de `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

## Fixtures

Las pruebas usan fixtures de pytest para configurar el entorno de prueba. Los fixtures comunes incluyen:

- `mock_settings`: Mock de Settings con valores de prueba
- `mock_gemini`: Mock de GeminiProvider
- `mock_repository`: Mock de CorpusRepository
- `mock_connection`: Mock de conexión a base de datos

## Convenciones

1. **Nomenclatura**: Los archivos de prueba deben nombrarse como `test_<módulo>.py`
2. **Funciones de prueba**: Deben comenzar con `test_`
3. **Docstrings**: Cada prueba debe tener un docstring descriptivo
4. **Mocks**: Usar `unittest.mock` para aislar dependencias externas
5. **Assertions**: Usar assertions claros y específicos

## Agregar Nuevas Pruebas

1. Crear un archivo `test_<módulo>.py` en el directorio `tests/`
2. Importar pytest y los módulos a probar
3. Crear fixtures si es necesario
4. Escribir funciones de prueba con el prefijo `test_`
5. Ejecutar las pruebas para verificar que pasan

## Cobertura

Objetivo de cobertura: >80% para el código principal.

Para ver el reporte de cobertura:
```bash
pytest --cov=corpus_ai --cov-report=html
open htmlcov/index.html
```

## CI/CD

Las pruebas se ejecutan automáticamente en CI/CD. Asegúrate de que todas las pruebas pasen antes de hacer push.
