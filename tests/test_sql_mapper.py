import pytest

from corpus_ai.ingestion.sql_mapper import SQLTableMapping


def test_sql_table_mapping_creation():
    """Test SQLTableMapping dataclass creation"""
    mapping = SQLTableMapping(
        table_name="products",
        id_column="product_id",
        text_columns=["name", "description"],
        metadata_columns=["category", "price"]
    )
    
    assert mapping.table_name == "products"
    assert mapping.id_column == "product_id"
    assert mapping.text_columns == ["name", "description"]
    assert mapping.metadata_columns == ["category", "price"]
    assert mapping.where_clause == "1=1"


def test_sql_table_mapping_with_custom_where():
    """Test SQLTableMapping with custom where clause"""
    mapping = SQLTableMapping(
        table_name="orders",
        id_column="order_id",
        text_columns=["order_details"],
        metadata_columns=["status"],
        where_clause="status = 'active'"
    )
    
    assert mapping.where_clause == "status = 'active'"


def test_sql_table_mapping_empty_lists():
    """Test SQLTableMapping with empty lists"""
    mapping = SQLTableMapping(
        table_name="test",
        id_column="id",
        text_columns=[],
        metadata_columns=[]
    )
    
    assert mapping.text_columns == []
    assert mapping.metadata_columns == []


def test_sql_table_mapping_single_column():
    """Test SQLTableMapping with single text column"""
    mapping = SQLTableMapping(
        table_name="users",
        id_column="user_id",
        text_columns=["bio"],
        metadata_columns=[]
    )
    
    assert len(mapping.text_columns) == 1
    assert mapping.text_columns[0] == "bio"
