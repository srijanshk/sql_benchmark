"""
Gold Set Evaluator for Schema Linking

Evaluates schema linking performance against gold annotations from spider2-snow-gold-tables.jsonl.
Calculates Recall@K, Precision@K, MRR, and provides detailed error analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time

from ..schema_linker import LinkerResult, TableCandidate, SchemaLinker
from ..embedding_linker import OllamaEmbeddingLinker
from ..metadata_loader import DatabaseMetadata


logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Metrics for a single evaluation instance."""
    instance_id: str
    question: str
    database: str
    gold_tables: Set[str]
    predicted_tables: List[str]
    
    # Metrics
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    f1_at_k: Dict[int, float] = field(default_factory=dict)
    average_precision: float = 0.0  # MAP component
    mrr: float = 0.0
    first_hit_rank: Optional[int] = None
    
    # Diagnostic info
    true_positives: Set[str] = field(default_factory=set)
    false_positives: Set[str] = field(default_factory=set)
    false_negatives: Set[str] = field(default_factory=set)
    
    def calculate_metrics(self, k_values: List[int] = [1, 3, 5, 10]):
        """Calculate all metrics."""
        # Find first hit for MRR
        for rank, table in enumerate(self.predicted_tables, start=1):
            if table in self.gold_tables:
                self.first_hit_rank = rank
                self.mrr = 1.0 / rank
                break
        
        # Calculate Recall@K, Precision@K, and F1@K
        for k in k_values:
            top_k = set(self.predicted_tables[:k])
            
            # True positives at this K
            tp = top_k & self.gold_tables
            
            # Recall@K = TP / total gold tables
            self.recall_at_k[k] = len(tp) / len(self.gold_tables) if self.gold_tables else 0.0
            
            # Precision@K = TP / K
            self.precision_at_k[k] = len(tp) / k if k > 0 else 0.0
            
            # F1@K = harmonic mean of precision and recall
            p = self.precision_at_k[k]
            r = self.recall_at_k[k]
            self.f1_at_k[k] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        # Calculate Average Precision (for MAP)
        # AP = sum of (precision@k * relevance@k) / total_relevant
        if self.gold_tables:
            precisions_at_hits = []
            hits_so_far = 0
            for rank, table in enumerate(self.predicted_tables, start=1):
                if table in self.gold_tables:
                    hits_so_far += 1
                    precision_at_rank = hits_so_far / rank
                    precisions_at_hits.append(precision_at_rank)
            
            self.average_precision = sum(precisions_at_hits) / len(self.gold_tables) if precisions_at_hits else 0.0
        
        # Calculate TP/FP/FN for error analysis
        predicted_set = set(self.predicted_tables[:max(k_values)])
        self.true_positives = predicted_set & self.gold_tables
        self.false_positives = predicted_set - self.gold_tables
        self.false_negatives = self.gold_tables - predicted_set


@dataclass
class AggregateMetrics:
    """Aggregate metrics across all instances."""
    total_instances: int = 0
    
    # Average metrics
    avg_recall_at_k: Dict[int, float] = field(default_factory=dict)
    avg_precision_at_k: Dict[int, float] = field(default_factory=dict)
    avg_f1_at_k: Dict[int, float] = field(default_factory=dict)
    mean_average_precision: float = 0.0  # MAP
    avg_mrr: float = 0.0
    
    # Coverage metrics
    perfect_recall_count: int = 0  # instances with recall@K = 1.0
    no_hit_count: int = 0  # instances with no gold table in top-K
    
    # Timing
    avg_latency_ms: float = 0.0
    
    def calculate_from_instances(self, instances: List[EvaluationMetrics], k_values: List[int]):
        """Aggregate metrics from individual instances."""
        self.total_instances = len(instances)
        
        if self.total_instances == 0:
            return
        
        # Initialize aggregators
        recall_sums = defaultdict(float)
        precision_sums = defaultdict(float)
        f1_sums = defaultdict(float)
        ap_sum = 0.0
        mrr_sum = 0.0
        
        for instance in instances:
            # Recall@K, Precision@K, F1@K
            for k in k_values:
                recall_sums[k] += instance.recall_at_k.get(k, 0.0)
                precision_sums[k] += instance.precision_at_k.get(k, 0.0)
                f1_sums[k] += instance.f1_at_k.get(k, 0.0)
            
            # AP and MRR
            ap_sum += instance.average_precision
            mrr_sum += instance.mrr
            
            # Perfect recall at max K
            max_k = max(k_values)
            if instance.recall_at_k.get(max_k, 0.0) >= 1.0:
                self.perfect_recall_count += 1
            
            # No hit
            if instance.first_hit_rank is None:
                self.no_hit_count += 1
        
        # Calculate averages
        for k in k_values:
            self.avg_recall_at_k[k] = recall_sums[k] / self.total_instances
            self.avg_precision_at_k[k] = precision_sums[k] / self.total_instances
            self.avg_f1_at_k[k] = f1_sums[k] / self.total_instances
        
        self.mean_average_precision = ap_sum / self.total_instances
        self.avg_mrr = mrr_sum / self.total_instances


class GoldSetEvaluator:
    """Evaluates schema linking against gold annotations."""
    
    def __init__(
        self,
        metadata: Dict[str, DatabaseMetadata],
        gold_file: Path,
        questions_file: Path,
        linker_type: str = "ollama",
        k_values: List[int] = [1, 3, 5, 10, 20]
    ):
        """
        Initialize evaluator.
        
        Args:
            metadata: Database metadata dict
            gold_file: Path to spider2-snow-gold-tables.jsonl
            questions_file: Path to spider2-snow.jsonl
            linker_type: Type of linker ("ollama" or "baseline")
            k_values: K values for Recall@K and Precision@K
        """
        self.metadata = metadata
        self.gold_file = gold_file
        self.questions_file = questions_file
        self.linker_type = linker_type
        self.k_values = sorted(k_values)
        
        # Load data
        self.gold_annotations = self._load_gold_annotations()
        self.questions = self._load_questions()
        
        # Linkers will be created per-database as needed
        self.linkers: Dict[str, SchemaLinker] = {}
    
    def _get_linker(self, database_id: str) -> SchemaLinker:
        """Get or create linker for a specific database."""
        if database_id not in self.linkers:
            if database_id not in self.metadata:
                raise ValueError(f"No metadata found for database: {database_id}")
            
            db_metadata = self.metadata[database_id]
            
            if self.linker_type == "ollama":
                self.linkers[database_id] = OllamaEmbeddingLinker(
                    metadata=db_metadata
                )
            else:
                # Baseline linker needs just metadata
                from ..graph_builder import build_graph_from_metadata
                graph = build_graph_from_metadata(db_metadata)
                self.linkers[database_id] = SchemaLinker(graph)
        
        return self.linkers[database_id]
    
    def _load_gold_annotations(self) -> Dict[str, List[str]]:
        """Load gold table annotations."""
        annotations = {}
        with open(self.gold_file) as f:
            for line in f:
                data = json.loads(line)
                instance_id = data["instance_id"]
                gold_tables = data["gold_tables"]
                annotations[instance_id] = gold_tables
        
        logger.info(f"Loaded {len(annotations)} gold annotations")
        return annotations
    
    def _load_questions(self) -> Dict[str, Tuple[str, str]]:
        """Load questions and database IDs."""
        questions = {}
        with open(self.questions_file) as f:
            for line in f:
                data = json.loads(line)
                instance_id = data["instance_id"]
                question = data["instruction"]
                db_id = data["db_id"]
                questions[instance_id] = (question, db_id)
        
        logger.info(f"Loaded {len(questions)} questions")
        return questions
    
    def evaluate_instance(
        self,
        instance_id: str,
        max_k: Optional[int] = None
    ) -> Optional[EvaluationMetrics]:
        """
        Evaluate a single instance.
        
        Args:
            instance_id: Instance ID to evaluate
            max_k: Maximum K for retrieval (default: max of k_values)
            
        Returns:
            EvaluationMetrics or None if instance not found
        """
        if instance_id not in self.gold_annotations:
            logger.warning(f"No gold annotation for {instance_id}")
            return None
        
        if instance_id not in self.questions:
            logger.warning(f"No question for {instance_id}")
            return None
        
        question, db_id = self.questions[instance_id]
        gold_tables = self.gold_annotations[instance_id]
        
        if db_id not in self.metadata:
            logger.warning(f"No metadata for database {db_id}")
            return None
        
        # Run schema linking
        max_k = max_k or max(self.k_values)
        
        try:
            linker = self._get_linker(db_id)
            result: LinkerResult = linker.link(
                question=question,
                top_k=max_k
            )
            
            # Extract predicted tables
            predicted_tables = [tc.table for tc in result.candidate_tables]
            
            # Create metrics object
            metrics = EvaluationMetrics(
                instance_id=instance_id,
                question=question,
                database=db_id,
                gold_tables=set(gold_tables),
                predicted_tables=predicted_tables
            )
            
            # Calculate metrics
            metrics.calculate_metrics(self.k_values)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {instance_id}: {e}", exc_info=True)
            return None
    
    def evaluate_all(
        self,
        instance_ids: Optional[List[str]] = None,
        max_k: Optional[int] = None,
        save_results: bool = True,
        output_dir: Optional[Path] = None
    ) -> Tuple[List[EvaluationMetrics], AggregateMetrics]:
        """
        Evaluate all instances.
        
        Args:
            instance_ids: Specific instance IDs to evaluate (None = all)
            max_k: Maximum K for retrieval
            save_results: Whether to save detailed results
            output_dir: Directory for output files
            
        Returns:
            (instance_metrics, aggregate_metrics)
        """
        if instance_ids is None:
            instance_ids = list(self.gold_annotations.keys())
        
        logger.info(f"Evaluating {len(instance_ids)} instances...")
        
        instance_metrics = []
        total_time = 0.0
        
        for i, instance_id in enumerate(instance_ids, 1):
            logger.info(f"[{i}/{len(instance_ids)}] Evaluating {instance_id}")
            
            start_time = time.time()
            metrics = self.evaluate_instance(instance_id, max_k)
            elapsed = time.time() - start_time
            
            if metrics:
                instance_metrics.append(metrics)
                total_time += elapsed
                
                # Log progress
                if i % 10 == 0 or i == len(instance_ids):
                    avg_time = total_time / len(instance_metrics)
                    logger.info(
                        f"Progress: {i}/{len(instance_ids)} | "
                        f"Avg latency: {avg_time*1000:.1f}ms"
                    )
        
        # Calculate aggregate metrics
        aggregate = AggregateMetrics()
        aggregate.calculate_from_instances(instance_metrics, self.k_values)
        aggregate.avg_latency_ms = (total_time / len(instance_metrics) * 1000) if instance_metrics else 0.0
        
        # Save results
        if save_results:
            output_dir = output_dir or Path("evaluation_results")
            output_dir.mkdir(exist_ok=True)
            self._save_results(instance_metrics, aggregate, output_dir)
        
        return instance_metrics, aggregate
    
    def _save_results(
        self,
        instance_metrics: List[EvaluationMetrics],
        aggregate: AggregateMetrics,
        output_dir: Path
    ):
        """Save evaluation results to files."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save aggregate metrics
        aggregate_file = output_dir / f"aggregate_metrics_{timestamp}.json"
        with open(aggregate_file, 'w') as f:
            json.dump({
                "total_instances": aggregate.total_instances,
                "avg_recall_at_k": aggregate.avg_recall_at_k,
                "avg_precision_at_k": aggregate.avg_precision_at_k,
                "avg_mrr": aggregate.avg_mrr,
                "perfect_recall_count": aggregate.perfect_recall_count,
                "no_hit_count": aggregate.no_hit_count,
                "avg_latency_ms": aggregate.avg_latency_ms
            }, f, indent=2)
        logger.info(f"Saved aggregate metrics to {aggregate_file}")
        
        # Save detailed instance results
        instance_file = output_dir / f"instance_results_{timestamp}.jsonl"
        with open(instance_file, 'w') as f:
            for metrics in instance_metrics:
                result = {
                    "instance_id": metrics.instance_id,
                    "question": metrics.question,
                    "database": metrics.database,
                    "gold_tables": list(metrics.gold_tables),
                    "predicted_tables": metrics.predicted_tables,
                    "recall_at_k": metrics.recall_at_k,
                    "precision_at_k": metrics.precision_at_k,
                    "mrr": metrics.mrr,
                    "first_hit_rank": metrics.first_hit_rank,
                    "true_positives": list(metrics.true_positives),
                    "false_positives": list(metrics.false_positives),
                    "false_negatives": list(metrics.false_negatives)
                }
                f.write(json.dumps(result) + '\n')
        logger.info(f"Saved instance results to {instance_file}")
        
        # Generate error analysis report
        self._generate_error_report(instance_metrics, output_dir, timestamp)
    
    def _generate_error_report(
        self,
        instance_metrics: List[EvaluationMetrics],
        output_dir: Path,
        timestamp: str
    ):
        """Generate detailed error analysis report."""
        report_file = output_dir / f"error_analysis_{timestamp}.md"
        
        # Analyze errors
        no_hits = [m for m in instance_metrics if m.first_hit_rank is None]
        partial_recalls = [m for m in instance_metrics if 0 < m.recall_at_k.get(10, 0) < 1.0]
        perfect_recalls = [m for m in instance_metrics if m.recall_at_k.get(10, 0) >= 1.0]
        
        # Database-level breakdown
        db_performance = defaultdict(list)
        for m in instance_metrics:
            db_performance[m.database].append(m.recall_at_k.get(10, 0))
        
        with open(report_file, 'w') as f:
            f.write("# Schema Linking Error Analysis\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall stats
            f.write("## Overall Statistics\n\n")
            f.write(f"- Total instances: {len(instance_metrics)}\n")
            if len(instance_metrics) > 0:
                f.write(f"- Perfect recalls (R@10=1.0): {len(perfect_recalls)} ({len(perfect_recalls)/len(instance_metrics)*100:.1f}%)\n")
                f.write(f"- Partial recalls (0 < R@10 < 1.0): {len(partial_recalls)} ({len(partial_recalls)/len(instance_metrics)*100:.1f}%)\n")
                f.write(f"- No hits (R@10=0.0): {len(no_hits)} ({len(no_hits)/len(instance_metrics)*100:.1f}%)\n\n")
            else:
                f.write("- No instances were successfully evaluated\n\n")
            
            # Database breakdown
            f.write("## Performance by Database\n\n")
            f.write("| Database | Instances | Avg Recall@10 |\n")
            f.write("|----------|-----------|---------------|\n")
            for db, recalls in sorted(db_performance.items(), key=lambda x: -sum(x[1])/len(x[1])):
                avg_recall = sum(recalls) / len(recalls)
                f.write(f"| {db} | {len(recalls)} | {avg_recall:.3f} |\n")
            f.write("\n")
            
            # Failure cases
            if no_hits:
                f.write(f"## Complete Failures ({len(no_hits)} cases)\n\n")
                for m in no_hits[:20]:  # Show top 20
                    f.write(f"### {m.instance_id}\n")
                    f.write(f"**Question:** {m.question}\n\n")
                    f.write(f"**Database:** {m.database}\n\n")
                    f.write(f"**Gold tables:** {', '.join(m.gold_tables)}\n\n")
                    f.write(f"**Top-10 predictions:** {', '.join(m.predicted_tables[:10])}\n\n")
                    f.write("---\n\n")
            
            # Partial success cases
            if partial_recalls:
                f.write(f"## Partial Recalls ({len(partial_recalls)} cases)\n\n")
                for m in sorted(partial_recalls, key=lambda x: x.recall_at_k.get(10, 0))[:10]:
                    f.write(f"### {m.instance_id} (R@10={m.recall_at_k.get(10, 0):.2f})\n")
                    f.write(f"**Question:** {m.question}\n\n")
                    f.write(f"**True positives:** {', '.join(m.true_positives) if m.true_positives else 'None'}\n\n")
                    f.write(f"**False negatives:** {', '.join(m.false_negatives)}\n\n")
                    f.write(f"**False positives:** {', '.join(list(m.false_positives)[:5])}\n\n")
                    f.write("---\n\n")
        
        logger.info(f"Saved error analysis to {report_file}")
    
    def print_summary(self, aggregate: AggregateMetrics):
        """Print summary of aggregate metrics."""
        print("\n" + "="*80)
        print("SCHEMA LINKING EVALUATION RESULTS")
        print("="*80)
        print(f"\nTotal instances evaluated: {aggregate.total_instances}")
        print(f"Average latency: {aggregate.avg_latency_ms:.1f}ms")
        print(f"\n{'Metric':<20} {'Value':<10}")
        print("-" * 30)
        
        for k in sorted(aggregate.avg_recall_at_k.keys()):
            print(f"Recall@{k:<14} {aggregate.avg_recall_at_k[k]:.4f}")
        
        print()
        for k in sorted(aggregate.avg_precision_at_k.keys()):
            print(f"Precision@{k:<11} {aggregate.avg_precision_at_k[k]:.4f}")
        
        print()
        for k in sorted(aggregate.avg_f1_at_k.keys()):
            print(f"F1@{k:<17} {aggregate.avg_f1_at_k[k]:.4f}")
        
        print(f"\nMean Average Precision (MAP): {aggregate.mean_average_precision:.4f}")
        print(f"Mean Reciprocal Rank (MRR): {aggregate.avg_mrr:.4f}")
        print(f"\nPerfect recalls (R@{max(aggregate.avg_recall_at_k.keys())}=1.0): {aggregate.perfect_recall_count} ({aggregate.perfect_recall_count/aggregate.total_instances*100:.1f}%)")
        print(f"No hits: {aggregate.no_hit_count} ({aggregate.no_hit_count/aggregate.total_instances*100:.1f}%)")
        print("="*80 + "\n")
