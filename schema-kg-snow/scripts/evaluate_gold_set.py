#!/usr/bin/env python3
"""
Evaluate schema linking performance against Spider2-Snow gold set.

Usage:
    conda activate spider2 && python scripts/evaluate_gold_set.py --linker ollama --top-k 10
    conda activate spider2 && python scripts/evaluate_gold_set.py --linker ollama --instance-ids sf_bq001 sf_bq002
    conda activate spider2 && python scripts/evaluate_gold_set.py --linker baseline --sample 50

IMPORTANT: Always run with 'conda activate spider2 &&' prefix to ensure correct environment!
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Verify we're in the correct conda environment
if 'CONDA_DEFAULT_ENV' in os.environ and os.environ['CONDA_DEFAULT_ENV'] != 'spider2':
    print("WARNING: You're not in the 'spider2' conda environment!", file=sys.stderr)
    print("Run: conda activate spider2 && python scripts/evaluate_gold_set.py ...", file=sys.stderr)
    sys.exit(1)

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from schema_kg_snow.metadata_loader import load_metadata
from schema_kg_snow.evaluation import GoldSetEvaluator


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate schema linking against gold set")
    
    # Data paths
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data/metadata"),
        help="Directory with metadata JSON files"
    )
    parser.add_argument(
        "--gold-file",
        type=Path,
        default=Path("methods/gold-tables/spider2-snow-gold-tables.jsonl"),
        help="Path to gold annotations file"
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=Path("spider2-snow/spider2-snow.jsonl"),
        help="Path to questions file"
    )
    
    # Linker configuration
    parser.add_argument(
        "--linker",
        choices=["baseline", "ollama"],
        default="ollama",
        help="Linker type to evaluate"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Maximum K for retrieval"
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="K values for metrics (e.g., --k-values 1 3 5 10)"
    )
    
    # Evaluation scope
    parser.add_argument(
        "--instance-ids",
        nargs="+",
        help="Specific instance IDs to evaluate (default: all)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Random sample size (for quick testing)"
    )
    parser.add_argument(
        "--databases",
        nargs="+",
        help="Filter to specific databases"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results"),
        help="Directory for evaluation results"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save detailed results"
    )
    
    # Other
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # Validate paths
    if not args.gold_file.exists():
        logger.error(f"Gold file not found: {args.gold_file}")
        return 1
    
    if not args.questions_file.exists():
        logger.error(f"Questions file not found: {args.questions_file}")
        return 1
    
    # Load metadata
    logger.info("Loading database metadata...")
    metadata = load_metadata(args.metadata_dir)
    logger.info(f"Loaded metadata for {len(metadata)} databases")
    
    # Create evaluator
    logger.info(f"Creating evaluator with {args.linker} linker...")
    evaluator = GoldSetEvaluator(
        metadata=metadata,
        gold_file=args.gold_file,
        questions_file=args.questions_file,
        linker_type=args.linker,
        k_values=args.k_values
    )
    
    # Determine instance IDs to evaluate
    instance_ids = args.instance_ids
    
    if args.sample and not instance_ids:
        import random
        all_ids = list(evaluator.gold_annotations.keys())
        instance_ids = random.sample(all_ids, min(args.sample, len(all_ids)))
        logger.info(f"Randomly sampled {len(instance_ids)} instances")
    
    if args.databases:
        # Filter to specific databases
        filtered_ids = []
        for iid in (instance_ids or evaluator.gold_annotations.keys()):
            if iid in evaluator.questions:
                _, db_id = evaluator.questions[iid]
                if db_id in args.databases:
                    filtered_ids.append(iid)
        instance_ids = filtered_ids
        logger.info(f"Filtered to {len(instance_ids)} instances in databases: {args.databases}")
    
    # Run evaluation
    logger.info("Starting evaluation...")
    instance_metrics, aggregate = evaluator.evaluate_all(
        instance_ids=instance_ids,
        max_k=args.top_k,
        save_results=not args.no_save,
        output_dir=args.output_dir
    )
    
    # Print summary
    evaluator.print_summary(aggregate)
    
    # Additional analysis
    if instance_metrics:
        logger.info("\n--- Top 10 Best Performing Instances ---")
        sorted_by_recall = sorted(
            instance_metrics,
            key=lambda m: (m.recall_at_k.get(args.top_k, 0), m.mrr),
            reverse=True
        )
        for i, m in enumerate(sorted_by_recall[:10], 1):
            logger.info(
                f"{i}. {m.instance_id} | R@{args.top_k}={m.recall_at_k.get(args.top_k, 0):.3f} | "
                f"MRR={m.mrr:.3f} | {m.database}"
            )
        
        logger.info("\n--- Top 10 Worst Performing Instances ---")
        for i, m in enumerate(sorted_by_recall[-10:], 1):
            logger.info(
                f"{i}. {m.instance_id} | R@{args.top_k}={m.recall_at_k.get(args.top_k, 0):.3f} | "
                f"MRR={m.mrr:.3f} | {m.database}"
            )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
