"""Load schema metadata from Spider2-Snow resources."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ColumnMetadata:
    """Column-level metadata captured from the resource bundle."""

    name: str
    data_type: str
    description: Optional[str] = None
    sample_values: List[Any] = field(default_factory=list)


@dataclass
class TableMetadata:
    """Table or view metadata."""

    database: str
    schema: str
    name: str
    object_type: str = "table"  # table, view, etc.
    description: Optional[str] = None
    ddl: Optional[str] = None
    columns: List[ColumnMetadata] = field(default_factory=list)
    sample_rows: List[Dict[str, Any]] = field(default_factory=list)
    resource_path: Optional[Path] = None


@dataclass
class SchemaMetadata:
    """Schema container inside a Snowflake database."""

    database: str
    name: str
    tables: Dict[str, TableMetadata] = field(default_factory=dict)


@dataclass
class DatabaseMetadata:
    """Top-level database metadata for graph building."""

    name: str
    schemas: Dict[str, SchemaMetadata] = field(default_factory=dict)
    resource_path: Optional[Path] = None

    @property
    def table_count(self) -> int:
        return sum(len(schema.tables) for schema in self.schemas.values())

    @property
    def column_count(self) -> int:
        return sum(len(table.columns) for schema in self.schemas.values() for table in schema.tables.values())


def discover_databases(resource_root: Path) -> List[str]:
    """List database names that have resource folders."""

    if not resource_root.exists():
        raise FileNotFoundError(f"Resource root not found: {resource_root}")

    databases = sorted(
        entry.name
        for entry in resource_root.iterdir()
        if entry.is_dir() and not entry.name.startswith(".")
    )
    return databases


def load_database_metadata(resource_root: Path, database_name: str) -> DatabaseMetadata:
    """Load a database's schema metadata from the Spider2-Snow bundle."""

    db_dir = resource_root / database_name
    if not db_dir.exists():
        raise FileNotFoundError(f"Database '{database_name}' not found under {resource_root}")

    database = DatabaseMetadata(name=database_name, resource_path=db_dir)

    # Traverse schema directories under the database folder
    for schema_dir in sorted(path for path in db_dir.glob("**/*") if path.is_dir()):
        # Skip the database root itself; schema dirs contain JSON docs
        if schema_dir == db_dir:
            continue

        schema_name = schema_dir.name.upper()
        schema_meta = SchemaMetadata(database=database_name, name=schema_name)

        ddl_map = _load_schema_ddl(schema_dir)

        # Each JSON file describes a table or view
        for json_path in sorted(schema_dir.glob("*.json")):
            table_meta = _parse_table_json(json_path, database_name, schema_name, ddl_map)
            if table_meta:
                schema_meta.tables[table_meta.name.upper()] = table_meta

        if schema_meta.tables:
            database.schemas[schema_name] = schema_meta

    if not database.schemas:
        raise ValueError(f"No schema metadata found for database '{database_name}' in {db_dir}")

    return database


def _load_schema_ddl(schema_dir: Path) -> Dict[str, str]:
    """Load the schema's DDL.csv file into a lookup map."""

    ddl_path = schema_dir / "DDL.csv"
    if not ddl_path.exists():
        return {}

    ddl_map: Dict[str, str] = {}
    with ddl_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            table_name = (row.get("table_name") or "").strip().upper()
            ddl = row.get("DDL") or row.get("ddl")
            if table_name and ddl:
                ddl_map[table_name] = ddl
    return ddl_map


def _parse_table_json(
    json_path: Path,
    fallback_database: str,
    fallback_schema: str,
    ddl_map: Dict[str, str],
) -> Optional[TableMetadata]:
    """Parse a per-table JSON file."""

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON at {json_path}") from exc

    table_fullname = data.get("table_fullname") or ""
    db_name, schema_name, table_name = _infer_table_identity(
        table_fullname,
        data.get("table_name"),
        fallback_database,
        fallback_schema,
    )

    column_names = data.get("column_names", []) or []
    column_types = data.get("column_types", []) or []
    column_descriptions = data.get("description", []) or []
    sample_rows: List[Dict[str, Any]] = data.get("sample_rows") or []

    columns = []
    for idx, col_name in enumerate(column_names):
        col_type = column_types[idx] if idx < len(column_types) else ""
        col_desc = column_descriptions[idx] if idx < len(column_descriptions) else None
        sample_values = _extract_sample_values(sample_rows, col_name)
        columns.append(
            ColumnMetadata(
                name=col_name,
                data_type=col_type,
                description=col_desc,
                sample_values=sample_values,
            )
        )

    ddl_key = table_name.upper()
    ddl = ddl_map.get(ddl_key)

    table_meta = TableMetadata(
        database=db_name,
        schema=schema_name,
        name=table_name,
        object_type=data.get("table_type") or data.get("type") or "table",
        description=data.get("table_description") or data.get("description_text"),
        ddl=ddl,
        columns=columns,
        sample_rows=sample_rows,
        resource_path=json_path,
    )
    return table_meta


def _infer_table_identity(
    table_fullname: str,
    table_name_field: Optional[str],
    fallback_database: str,
    fallback_schema: str,
) -> Tuple[str, str, str]:
    """Infer database, schema, and table names from JSON payload."""

    parts: List[str] = []
    if table_fullname:
        parts = [p for p in table_fullname.split(".") if p]
    elif table_name_field and "." in table_name_field:
        parts = [p for p in table_name_field.split(".") if p]

    if len(parts) >= 3:
        db_name, schema_name, table_name = parts[-3], parts[-2], parts[-1]
    elif len(parts) == 2:
        db_name, schema_name, table_name = parts[0], parts[0], parts[1]
    else:
        db_name = fallback_database
        schema_name = fallback_schema
        table_name = table_name_field or "UNKNOWN"

    return db_name.upper(), schema_name.upper(), table_name.upper()


def _extract_sample_values(sample_rows: List[Dict[str, Any]], column_name: str) -> List[Any]:
    """Collect sample values for a column from the sample rows."""

    values: List[Any] = []
    for row in sample_rows:
        if isinstance(row, dict) and column_name in row:
            values.append(row[column_name])
    # Limit to a few examples to keep metadata compact
    return values[:5]
