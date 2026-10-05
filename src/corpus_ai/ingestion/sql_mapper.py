# src/corpus_ai/ingestion/sql_mapper.py
from dataclasses import dataclass
from typing import List

@dataclass
class SQLTableMapping:
    table_name: str
    id_column: str
    text_columns: List[str]  # Columnas que se concatenan para formar el 'text'
    metadata_columns: List[str]  # Columnas que van al diccionario 'metadata'
    where_clause: str = "1=1"