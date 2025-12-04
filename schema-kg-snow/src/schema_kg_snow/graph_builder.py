"""Build NetworkX graphs from Snowflake metadata."""

from __future__ import annotations

from typing import Dict

import networkx as nx

from .metadata_loader import DatabaseMetadata


def build_graph_from_metadata(
    database: DatabaseMetadata, concept_model: str = "llama3"
) -> nx.DiGraph:
    """Create a directed graph for a single database."""

    G = nx.DiGraph()

    db_node = f"database:{database.name}"
    G.add_node(
        db_node,
        kind="database",
        name=database.name,
        resource_path=str(database.resource_path) if database.resource_path else None,
    )

    from .concept_extractor import extract_concepts_from_table

    for schema in database.schemas.values():
        schema_node = f"schema:{database.name}.{schema.name}"
        G.add_node(
            schema_node,
            kind="schema",
            name=schema.name,
            database=database.name,
        )
        G.add_edge(db_node, schema_node, rel="HAS_SCHEMA")

        for table in schema.tables.values():
            table_node = f"table:{database.name}.{schema.name}.{table.name}"
            G.add_node(
                table_node,
                kind="table",
                name=table.name,
                database=database.name,
                schema=schema.name,
                object_type=table.object_type,
                description=table.description,
                ddl=table.ddl,
                resource_path=str(table.resource_path) if table.resource_path else None,
            )
            G.add_edge(schema_node, table_node, rel="HAS_TABLE")

            # Extract and add concepts
            concepts = extract_concepts_from_table(table, model=concept_model)
            for concept in concepts:
                concept_node = f"concept:{concept.lower().replace(' ', '_')}"
                if concept_node not in G:
                    G.add_node(concept_node, kind="concept", name=concept)
                G.add_edge(table_node, concept_node, rel="RELATED_TO")

            for column in table.columns:
                column_node = f"column:{database.name}.{schema.name}.{table.name}.{column.name}"
                G.add_node(
                    column_node,
                    kind="column",
                    name=column.name,
                    database=database.name,
                    schema=schema.name,
                    table=table.name,
                    data_type=column.data_type,
                    description=column.description,
                    sample_values=column.sample_values,
                )
                G.add_edge(table_node, column_node, rel="HAS_COLUMN")

    # Add foreign key edges by parsing DDL
    _add_foreign_key_edges(G, database)

    return G


def _add_foreign_key_edges(G: nx.DiGraph, database: DatabaseMetadata) -> None:
    """
    Parse DDL statements to extract foreign key relationships and add FK edges.
    
    Args:
        G: NetworkX graph to add edges to
        database: Database metadata containing DDL statements
    """
    import re
    
    for schema in database.schemas.values():
        for table in schema.tables.values():
            if not table.ddl:
                continue
            
            # Parse FK constraints from DDL
            # Pattern: FOREIGN KEY (column) REFERENCES target_table(target_column)
            fk_pattern = r'FOREIGN\s+KEY\s*\([^)]+\)\s+REFERENCES\s+([^\s(]+)'
            
            matches = re.findall(fk_pattern, table.ddl, re.IGNORECASE)
            
            for ref_table_name in matches:
                # Clean table name (remove quotes, schema prefix)
                ref_table_name = ref_table_name.strip('"').strip("'")
                if '.' in ref_table_name:
                    ref_table_name = ref_table_name.split('.')[-1]
                
                # Build source and target node IDs
                source_table = f"table:{database.name}.{schema.name}.{table.name}"
                
                # Try to find target table in same schema first, then other schemas
                target_table = None
                for s in database.schemas.values():
                    if ref_table_name.upper() in [t.name.upper() for t in s.tables.values()]:
                        # Find exact match
                        for t in s.tables.values():
                            if t.name.upper() == ref_table_name.upper():
                                target_table = f"table:{database.name}.{s.name}.{t.name}"
                                break
                        if target_table:
                            break
                
                # Add FK edge if both nodes exist
                if target_table and source_table in G and target_table in G:
                    G.add_edge(source_table, target_table, rel="FOREIGN_KEY")


def summarize_graph(G: nx.DiGraph) -> Dict[str, int]:
    """Quick counts for graph contents."""

    summary = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "databases": sum(1 for _, attrs in G.nodes(data=True) if attrs.get("kind") == "database"),
        "schemas": sum(1 for _, attrs in G.nodes(data=True) if attrs.get("kind") == "schema"),
        "tables": sum(1 for _, attrs in G.nodes(data=True) if attrs.get("kind") == "table"),
        "columns": sum(1 for _, attrs in G.nodes(data=True) if attrs.get("kind") == "column"),
    }
    return summary
