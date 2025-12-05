"""Command line interface for schema-kg-snow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import networkx as nx
from networkx.readwrite import json_graph

from .metadata_loader import discover_databases, load_database_metadata
from .graph_builder import build_graph_from_metadata, summarize_graph
from .schema_linker import SchemaLinker
from .embedding_linker import OllamaEmbeddingLinker
from .exporter import export_schema_pack
from .visualize import visualize_graph_pyvis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Schema KG tooling for Spider2-Snow")
    parser.add_argument(
        "--resource-root",
        type=Path,
        default=Path("spider2-snow/resource/databases"),
        help="Path to the Spider2-Snow resource directory",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List available databases")

    inspect_parser = subparsers.add_parser("inspect", help="Inspect a database resource bundle")
    inspect_parser.add_argument("--database", required=True, help="Database name to inspect")

    graph_parser = subparsers.add_parser("build-graph", help="Build a graph for a database")
    graph_parser.add_argument("--database", required=True, help="Database name to graph")
    graph_parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to store the graph as JSON (node-link format)",
    )

    export_parser = subparsers.add_parser("export-pack", help="Export schema pack for agent usage")
    export_parser.add_argument("--database", required=True, help="Database name to export")
    export_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for the schema pack",
    )

    vis_parser = subparsers.add_parser("visualize", help="Render an interactive KG HTML via pyvis")
    vis_parser.add_argument("--database", required=True, help="Database name to visualize")
    vis_parser.add_argument(
        "--output",
        type=Path,
        default=Path("visualizations/kg.html"),
        help="Output HTML file",
    )
    link_parser = subparsers.add_parser("link", help="Run schema linker for a question")
    link_parser.add_argument("--database", required=True, help="Database name to query")
    link_parser.add_argument("--question", required=True, help="Natural language instruction")
    link_parser.add_argument("--top-k", type=int, default=5, help="How many candidates to show")
    link_parser.add_argument(
        "--linker",
        choices=["baseline", "ollama"],
        default="baseline",
        help="Choose between lexical linker or Ollama embedding linker",
    )
    link_parser.add_argument(
        "--embedding-model",
        default="nomic-embed-text",
        help="Embedding model name when using the Ollama linker",
    )
    link_parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("schema-kg-snow/cache"),
        help="Directory to store/reuse Ollama embedding caches",
    )
    link_parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force regeneration of embedding cache for the target database",
    )

    predict_parser = subparsers.add_parser("predict", help="Run inference without gold SQL")
    predict_parser.add_argument("--database", required=True, help="Database name")
    predict_parser.add_argument("--question", required=True, help="Natural language question")
    predict_parser.add_argument("--top-k", type=int, default=5, help="Number of candidates")
    predict_parser.add_argument("--concept-model", default="llama3", help="Ollama model for concepts")
    predict_parser.add_argument("--embedding-model", default="nomic-embed-text", help="Ollama model for embeddings")
    predict_parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("schema-kg-snow/cache"),
        help="Path to embedding cache",
    )

    return parser.parse_args()


def cmd_list(resource_root: Path) -> None:
    databases = discover_databases(resource_root)
    print(f"Found {len(databases)} databases under {resource_root}")
    for name in databases:
        print(f"- {name}")


def cmd_inspect(resource_root: Path, database: str) -> None:
    metadata = load_database_metadata(resource_root, database)
    print(f"Database: {metadata.name}")
    print(f"  Schemas: {len(metadata.schemas)}")
    print(f"  Tables: {metadata.table_count}")
    print(f"  Columns: {metadata.column_count}")
    for schema_name, schema in metadata.schemas.items():
        print(f"    - {schema_name}: {len(schema.tables)} tables")


def cmd_build_graph(resource_root: Path, database: str, output: Path | None) -> None:
    metadata = load_database_metadata(resource_root, database)
    graph = build_graph_from_metadata(metadata)
    summary = summarize_graph(graph)
    print(json.dumps(summary, indent=2))

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        serialized: Dict[str, Any] = json_graph.node_link_data(graph)
        with output.open("w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)
        print(f"Graph saved to {output}")


def cmd_export_pack(resource_root: Path, database: str, output_dir: Path) -> None:
    metadata = load_database_metadata(resource_root, database)
    outputs = export_schema_pack(metadata, output_dir)
    print(f"Schema pack exported to {output_dir}")
    for kind, path in outputs.items():
        print(f"  - {kind}: {path}")


def cmd_visualize(resource_root: Path, database: str, output: Path) -> None:
    metadata = load_database_metadata(resource_root, database)
    graph = build_graph_from_metadata(metadata)
    output.parent.mkdir(parents=True, exist_ok=True)
    visualize_graph_pyvis(graph, output)
    print(f"Visualization saved to {output}")


def cmd_link(
    resource_root: Path,
    database: str,
    question: str,
    top_k: int,
    linker_type: str,
    embedding_model: str,
    embedding_cache: Path,
    refresh_cache: bool,
) -> None:
    metadata = load_database_metadata(resource_root, database)

    if linker_type == "ollama":
        linker = OllamaEmbeddingLinker(
            metadata,
            embedding_model=embedding_model,
            cache_dir=embedding_cache,
            refresh_cache=refresh_cache,
        )
    else:
        linker = SchemaLinker.from_metadata(metadata)

    result = linker.link(question, top_k=top_k)

    print(f"\nQuestion: {question}")
    print(f"\n=== SCHEMA LINKING RESULTS ===\n")
    
    # Display concepts
    if result.concepts:
        print(f"Validated Concepts: {', '.join(result.concepts)}")
    
    # Display constraints
    if result.constraints:
        print(f"\nExtracted Constraints:")
        for constraint in result.constraints:
            print(f"  - {constraint['type']}: {constraint['operator']} {constraint['value']} (hint: {constraint.get('column_hint', 'N/A')})")
    
    # Display candidate tables
    print(f"\nCandidate Tables:")
    for table_candidate in result.candidate_tables:
        print(f"  {table_candidate.table} -> Score: {table_candidate.score:.3f}")
        if table_candidate.why:
            for reason in table_candidate.why:
                print(f"    • {reason}")

    # Display candidate columns
    print(f"\nCandidate Columns:")
    for col_candidate in result.candidate_columns:
        role_str = f" [{col_candidate.role}]" if col_candidate.role else ""
        print(f"  {col_candidate.column}{role_str} -> Score: {col_candidate.score:.3f}")
        if col_candidate.why:
            for reason in col_candidate.why:
                print(f"    • {reason}")
    
    # Display join paths
    if result.join_paths:
        print(f"\nDiscovered Join Paths:")
        for join_path in result.join_paths:
            tables_str = " → ".join(join_path.tables)
            print(f"  {tables_str} (Score: {join_path.score:.2f})")


def cmd_predict(
    resource_root: Path,
    database: str,
    question: str,
    top_k: int,
    concept_model: str,
    embedding_model: str,
    embedding_cache: Path,
) -> None:
    metadata = load_database_metadata(resource_root, database)
    
    print(f"Building graph for {database} using {concept_model} for concepts...")
    linker = OllamaEmbeddingLinker(
        metadata,
        embedding_model=embedding_model,
        concept_model=concept_model,
        cache_dir=embedding_cache,
    )

    print(f"Linking question: '{question}'")
    result = linker.link(question, top_k=top_k)

    print("\n=== SCHEMA LINKING RESULTS ===\n")
    
    print("Top Candidate Tables:")
    for table_candidate in result.candidate_tables:
        print(f"  - {table_candidate.table} (Score: {table_candidate.score:.4f})")
        
    print("\nTop Candidate Columns:")
    for col_candidate in result.candidate_columns:
        role_str = f" [{col_candidate.role}]" if col_candidate.role else ""
        print(f"  - {col_candidate.column}{role_str} (Score: {col_candidate.score:.4f})")
    
    if result.join_paths:
        print("\nJoin Paths:")
        for join_path in result.join_paths:
            print(f"  - {' → '.join(join_path.tables)}")

def main() -> None:
    args = parse_args()
    resource_root: Path = args.resource_root

    if args.command == "list":
        cmd_list(resource_root)
    elif args.command == "inspect":
        cmd_inspect(resource_root, args.database)
    elif args.command == "build-graph":
        cmd_build_graph(resource_root, args.database, args.output)
    elif args.command == "export-pack":
        cmd_export_pack(resource_root, args.database, args.output_dir)
    elif args.command == "visualize":
        cmd_visualize(resource_root, args.database, args.output)
    elif args.command == "link":
        cmd_link(
            resource_root,
            args.database,
            args.question,
            args.top_k,
            linker_type=args.linker,
            embedding_model=args.embedding_model,
            embedding_cache=args.embedding_cache,
            refresh_cache=args.refresh_cache,
        )
    elif args.command == "predict":
        cmd_predict(
            resource_root,
            args.database,
            args.question,
            args.top_k,
            args.concept_model,
            args.embedding_model,
            args.embedding_cache,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
