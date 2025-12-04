"""Evaluate schema linker recall for Spider2-Snow."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Dict, List, Optional

from schema_kg_snow import (
    discover_databases,
    load_database_metadata,
    build_graph_from_metadata,
)
from schema_kg_snow.schema_linker import SchemaLinker
from schema_kg_snow.embedding_linker import OllamaEmbeddingLinker
from schema_kg_snow.ollama_utils import check_ollama_connection


def load_tasks(jsonl_path: Path) -> List[Dict]:
    tasks = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    return tasks


def parse_gold_tables(sql_path: Path) -> List[str]:
    tables: List[str] = []
    if not sql_path.exists():
        return tables
    with sql_path.open("r", encoding="utf-8") as f:
        sql_text = f.read()

    fq_pattern = re.compile(r'([A-Za-z0-9_"]+)\.([A-Za-z0-9_"]+)\.([A-Za-z0-9_"]+)')
    for match in fq_pattern.findall(sql_text):
        table = match[2].replace('"', "").upper()
        tables.append(table)

    if tables:
        return sorted(set(tables))

    # fallback: look for schema.table in case db omitted
    short_pattern = re.compile(r'([A-Za-z0-9_"]+)\.([A-Za-z0-9_"]+)')
    for match in short_pattern.findall(sql_text):
        table = match[1].replace('"', "").upper()
        tables.append(table)

    return sorted(set(tables))


def evaluate(
    resource_root: Path,
    tasks_path: Path,
    sql_dir: Path,
    top_k: int = 5,
    use_ollama: bool = False,
    embedding_cache: Path | None = None,
    refresh_cache: bool = False,
    embedding_model: str = "nomic-embed-text",
    concept_model: str = "llama3",
) -> None:
    tasks = load_tasks(tasks_path)
    databases = discover_databases(resource_root)
    db_cache: Dict[str, SchemaLinker] = {}
    ollama_cache: Dict[str, OllamaEmbeddingLinker] = {}
    if use_ollama and not check_ollama_connection():
        print("[warn] Ollama linker requested but server is unavailable; falling back to baseline.")
        use_ollama = False

    recalls: List[float] = []
    evaluated_tasks = 0
    missing: List[str] = []

    for task in tasks:
        db_id = task["db_id"]
        if db_id not in databases:
            print(f"[skip] {task['instance_id']} -> database '{db_id}' not available")
            continue

        if use_ollama:
            if db_id not in ollama_cache:
                metadata = load_database_metadata(resource_root, db_id)
                ollama_cache[db_id] = OllamaEmbeddingLinker(
                    metadata,
                    cache_dir=embedding_cache,
                    refresh_cache=refresh_cache,
                    embedding_model=embedding_model,
                    concept_model=concept_model,
                )
            linker = ollama_cache[db_id]
        else:
            if db_id not in db_cache:
                metadata = load_database_metadata(resource_root, db_id)
                db_cache[db_id] = SchemaLinker.from_metadata(metadata)
            linker = db_cache[db_id]

        result = linker.link(task["instruction"], top_k=top_k)

        sql_gold = sql_dir / f"{task['instance_id']}.sql"
        if not sql_gold.exists():
            print(f"{task['instance_id']} | DB: {db_id}")
            print("  Gold SQL missing; skipping recall measurement\n")
            continue

        gold_tables = parse_gold_tables(sql_gold)

        predicted_tables = [node_id for node_id, _ in result.candidate_tables]
        matched = _match_tables(predicted_tables, gold_tables)

        recall = len(matched) / len(gold_tables) if gold_tables else 0.0

        print(f"{task['instance_id']} | DB: {db_id}")
        print(f"  Question: {task['instruction'][:80]}...")
        print(f"  Gold tables: {gold_tables}")
        formatted_predictions = [
            _format_table_node(linker.graph, node_id)
            for node_id in predicted_tables[:top_k]
        ]
        print(f"  Predicted: {formatted_predictions}")

        print(f"  Recall@{top_k}: {recall:.2f}\n")
        recalls.append(recall)
        evaluated_tasks += 1

        missing = [table for table in gold_tables if table not in matched]
        for table in missing:
            if not _table_exists(linker.graph, table):
                print(f"    - Missing '{table}': table not found in graph")
            else:
                print(f"    - Missing '{table}': in graph but not in top-{top_k}")
    if missing:
        print()

    if recalls:
        avg_recall = sum(recalls) / len(recalls)
        print("=" * 60)
        print(f"Evaluation summary ({'Ollama' if use_ollama else 'Baseline'} linker)")
        print(f"  Tasks evaluated: {evaluated_tasks}")
        print(f"  Average Recall@{top_k}: {avg_recall:.2f}")
        print("=" * 60)


def _match_tables(predicted_nodes: List[str], gold_tables: List[str]) -> List[str]:
    matched: set[str] = set()
    gold_set = {table.upper() for table in gold_tables}
    for node_id in predicted_nodes:
        _, fq_name = node_id.split(":", 1)
        table_name = fq_name.split(".")[-1].upper()
        if table_name in gold_set:
            matched.add(table_name)
    return list(matched)


def _format_table_node(graph, node_id: str) -> str:
    if node_id not in graph:
        return node_id
    attrs = graph.nodes[node_id]
    return f"{attrs.get('name')} ({node_id})"


def _table_exists(graph, table_name: str) -> bool:
    target = table_name.upper()
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("kind") == "table" and attrs.get("name", "").upper() == target:
            return True
    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate schema linker recall.")
    parser.add_argument("--resource-root", type=Path, default=Path("spider2-snow/resource/databases"))
    parser.add_argument("--tasks", type=Path, default=Path("spider2-snow/spider2-snow.jsonl"))
    parser.add_argument("--sql-dir", type=Path, default=Path("spider2-snow/evaluation_suite/gold/sql"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-ollama", action="store_true", help="Evaluate the Ollama embedding linker")
    parser.add_argument(
        "--embedding-model",
        default="nomic-embed-text",
        help="Embedding model name when using the Ollama linker",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path("schema-kg-snow/cache"),
        help="Directory for embedding cache (Ollama mode)",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force regeneration of embedding cache",
    )
    parser.add_argument(
        "--concept-model",
        default="llama3",
        help="Ollama model for concept extraction",
    )
    args = parser.parse_args()

    evaluate(
        resource_root=args.resource_root,
        tasks_path=args.tasks,
        sql_dir=args.sql_dir,
        top_k=args.top_k,
        use_ollama=args.use_ollama,
        embedding_cache=args.embedding_cache,
        refresh_cache=args.refresh_cache,
        embedding_model=args.embedding_model,
        concept_model=args.concept_model,
    )
