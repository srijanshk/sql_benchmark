#!/usr/bin/env python3
"""
Simple gold set evaluation test - bypasses numpy issues by using CLI directly.
"""

import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict


def load_gold_annotations(gold_file):
    """Load gold table annotations."""
    annotations = {}
    with open(gold_file) as f:
        for line in f:
            data = json.loads(line)
            annotations[data["instance_id"]] = data["gold_tables"]
    return annotations


def load_questions(questions_file):
    """Load questions."""
    questions = {}
    with open(questions_file) as f:
        for line in f:
            data = json.loads(line)
            questions[data["instance_id"]] = (data["instruction"], data["db_id"])
    return questions


def run_linker(database, question, top_k=10):
    """Run the schema linker via CLI and parse results."""
    cmd = [
        "python", "-m", "schema_kg_snow.cli",
        "link",
        "--database", database,
        "--question", question,
        "--linker", "ollama",
        "--top-k", str(top_k)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Parse output to extract predicted tables
        predicted_tables = []
        in_tables_section = False
        
        for line in result.stdout.split('\n'):
            if '[Step 7] Top tables:' in line:
                in_tables_section = True
                continue
            
            if in_tables_section:
                if line.strip().startswith('-'):
                    # Parse line like "- GA4.GA4_OBFUSCATED_SAMPLE_ECOMMERCE.EVENTS (0.342)"
                    parts = line.strip()[2:].split(' (')
                    if parts:
                        table = parts[0].strip()
                        predicted_tables.append(table)
                elif line.strip() and not line.strip().startswith('['):
                    # End of tables section
                    break
        
        return predicted_tables
        
    except Exception as e:
        print(f"Error running linker: {e}", file=sys.stderr)
        return []


def calculate_recall_at_k(gold_tables, predicted_tables, k):
    """Calculate Recall@K."""
    top_k = set(predicted_tables[:k])
    gold_set = set(gold_tables)
    
    if not gold_set:
        return 0.0
    
    hits = top_k & gold_set
    return len(hits) / len(gold_set)


def main():
    # Paths
    gold_file = Path("methods/gold-tables/spider2-snow-gold-tables.jsonl")
    questions_file = Path("spider2-snow/spider2-snow.jsonl")
    
    # Load data
    print("Loading annotations...")
    annotations = load_gold_annotations(gold_file)
    questions = load_questions(questions_file)
    
    # Sample for testing (take first 5)
    sample_ids = list(annotations.keys())[:5]
    
    print(f"\nEvaluating {len(sample_ids)} instances...\n")
    
    results = []
    for i, instance_id in enumerate(sample_ids, 1):
        if instance_id not in questions:
            continue
        
        question, db_id = questions[instance_id]
        gold_tables = annotations[instance_id]
        
        print(f"[{i}/{len(sample_ids)}] {instance_id}")
        print(f"  Database: {db_id}")
        print(f"  Gold tables: {gold_tables}")
        
        # Run linker
        predicted = run_linker(db_id, question, top_k=10)
        print(f"  Predicted (top-5): {predicted[:5]}")
        
        # Calculate metrics
        r1 = calculate_recall_at_k(gold_tables, predicted, 1)
        r5 = calculate_recall_at_k(gold_tables, predicted, 5)
        r10 = calculate_recall_at_k(gold_tables, predicted, 10)
        
        print(f"  Recall@1: {r1:.3f}, Recall@5: {r5:.3f}, Recall@10: {r10:.3f}")
        print()
        
        results.append({
            "instance_id": instance_id,
            "r1": r1,
            "r5": r5,
            "r10": r10,
            "gold": gold_tables,
            "predicted": predicted[:10]
        })
    
    # Aggregate
    if results:
        avg_r1 = sum(r["r1"] for r in results) / len(results)
        avg_r5 = sum(r["r5"] for r in results) / len(results)
        avg_r10 = sum(r["r10"] for r in results) / len(results)
        
        print("="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Instances evaluated: {len(results)}")
        print(f"Average Recall@1:  {avg_r1:.3f}")
        print(f"Average Recall@5:  {avg_r5:.3f}")
        print(f"Average Recall@10: {avg_r10:.3f}")
        print("="*80)


if __name__ == "__main__":
    main()
