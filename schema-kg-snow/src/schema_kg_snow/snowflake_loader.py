"""Load live Snowflake metadata into schema-kg data structures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import snowflake.connector  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    snowflake = None  # type: ignore
else:
    snowflake = snowflake.connector

from .metadata_loader import (
    ColumnMetadata,
    TableMetadata,
    SchemaMetadata,
    DatabaseMetadata,
)


def load_database_from_snowflake(
    credentials_path: Path,
    database: str,
    schema_filter: Optional[Iterable[str]] = None,
) -> DatabaseMetadata:
    """Fetch live metadata from Snowflake INFORMATION_SCHEMA."""

    if snowflake is None:
        raise ImportError("snowflake-connector-python is required for live metadata loading.")

    creds = _load_credentials(credentials_path)
    conn = snowflake.connect(
        user=creds["user"],
        password=creds["password"],
        account=creds["account"],
        warehouse=creds.get("warehouse"),
        role=creds.get("role"),
        database=database,
    )

    schema_filter_set = {s.upper() for s in schema_filter} if schema_filter else None
    db_metadata = DatabaseMetadata(name=database.upper())

    try:
        schemas = _fetch_schemas(conn, database)
        for schema_name in schemas:
            if schema_filter_set and schema_name.upper() not in schema_filter_set:
                continue
            schema_meta = SchemaMetadata(database=db_metadata.name, name=schema_name.upper())
            tables = _fetch_tables(conn, database, schema_name)
            for table in tables:
                columns = _fetch_columns(conn, database, schema_name, table)
                table_meta = TableMetadata(
                    database=db_metadata.name,
                    schema=schema_name.upper(),
                    name=table.upper(),
                    columns=columns,
                    ddl=None,
                    description=None,
                    object_type="table",
                )
                schema_meta.tables[table_meta.name] = table_meta
            if schema_meta.tables:
                db_metadata.schemas[schema_meta.name] = schema_meta
    finally:
        conn.close()

    return db_metadata


def _load_credentials(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    required = ["user", "password", "account"]
    for key in required:
        if key not in data:
            raise ValueError(f"Credential file missing '{key}'")
    return data


def _fetch_schemas(conn, database: str) -> List[str]:
    query = f"SHOW SCHEMAS IN DATABASE {database}"
    with conn.cursor() as cur:
        cur.execute(query)
        return [row[1] for row in cur.fetchall()]  # name column


def _fetch_tables(conn, database: str, schema: str) -> List[str]:
    query = f"SHOW TABLES IN SCHEMA {database}.{schema}"
    with conn.cursor() as cur:
        cur.execute(query)
        return [row[1] for row in cur.fetchall()]


def _fetch_columns(conn, database: str, schema: str, table: str) -> List[ColumnMetadata]:
    query = """
        SELECT column_name, data_type, comment
        FROM {db}.information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """.format(db=database)
    with conn.cursor() as cur:
        cur.execute(query, (schema, table))
        columns = []
        for name, data_type, comment in cur.fetchall():
            columns.append(
                ColumnMetadata(
                    name=name,
                    data_type=data_type,
                    description=comment,
                )
            )
        return columns
