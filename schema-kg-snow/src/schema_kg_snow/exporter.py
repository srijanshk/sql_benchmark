"""Utilities for exporting agent-friendly schema packs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from networkx.readwrite import json_graph

from .graph_builder import build_graph_from_metadata, summarize_graph
from .metadata_loader import DatabaseMetadata


def export_schema_pack(metadata: DatabaseMetadata, output_dir: Path) -> Dict[str, Path]:
    """Export graph + table summaries for an agent workspace."""

    output_dir.mkdir(parents=True, exist_ok=True)

    graph = build_graph_from_metadata(metadata)
    graph_path = output_dir / "graph.json"
    with graph_path.open("w", encoding="utf-8") as f:
        json.dump(json_graph.node_link_data(graph), f, indent=2)

    tables = []
    for schema in metadata.schemas.values():
        for table in schema.tables.values():
            tables.append(
                {
                    "database": table.database,
                    "schema": table.schema,
                    "name": table.name,
                    "object_type": table.object_type,
                    "description": table.description,
                    "ddl": table.ddl,
                    "columns": [
                        {
                            "name": col.name,
                            "data_type": col.data_type,
                            "description": col.description,
                            "sample_values": col.sample_values,
                        }
                        for col in table.columns
                    ],
                    "sample_rows": table.sample_rows[:20],
                }
            )

    tables_path = output_dir / "tables.json"
    with tables_path.open("w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2)

    summary = {
        "database": metadata.name,
        "schema_count": len(metadata.schemas),
        "table_count": metadata.table_count,
        "column_count": metadata.column_count,
        "graph_stats": summarize_graph(graph),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "graph": graph_path,
        "tables": tables_path,
        "summary": summary_path,
    }
