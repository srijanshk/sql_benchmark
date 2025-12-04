"""Graph RAG schema linker powered by Ollama embeddings."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

from .graph_builder import build_graph_from_metadata
from .metadata_loader import DatabaseMetadata
from .schema_linker import SchemaLinker
from .ollama_utils import embed_with_ollama, check_ollama_connection


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class OllamaEmbeddingLinker(SchemaLinker):
    """Schema linker that ranks tables with Ollama embeddings (Graph RAG style)."""

    def __init__(
        self,
        metadata: DatabaseMetadata,
        embedding_model: str = "nomic-embed-text",
        concept_model: str = "llama3",
        cache_dir: Path | None = None,
        refresh_cache: bool = False,
    ):
        graph = build_graph_from_metadata(metadata, concept_model=concept_model)
        super().__init__(graph)
        self.metadata = metadata
        self.embedding_model = embedding_model
        self.cache_dir = Path(cache_dir or "schema-kg-snow/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / f"{self.metadata.name}_embeddings.json"
        self.refresh_cache = refresh_cache
        self.ollama_available = check_ollama_connection()
        self.table_text_map = self._build_table_text_map()
        self.table_embeddings = self._precompute_table_embeddings()
        self.bm25_index, self.bm25_node_ids = self._build_bm25_index()

    def _build_bm25_index(self) -> Tuple[BM25Okapi | None, List[str]]:
        """Build BM25 index for lexical retrieval."""
        if not self.table_text_map:
            return None, []
        
        node_ids = list(self.table_text_map.keys())
        tokenized_corpus = [self._tokenize(text) for text in self.table_text_map.values()]
        bm25 = BM25Okapi(tokenized_corpus)
        return bm25, node_ids

    def _extract_query_concepts(self, question: str) -> str:
        """
        Extract key concepts (nouns, entities, values) from the question using LLM.
        This replaces hallucinated SQL generation with direct concept extraction.
        
        Args:
            question: Original user question
            
        Returns:
            String containing comma-separated concepts
        """
        from .ollama_utils import generate_with_ollama
        
        prompt = f'''Extract the key business concepts, entities, and values from the following question.
Focus on nouns, metrics, and specific filters.
Do NOT generate SQL. Do NOT guess table names.
Output ONLY a comma-separated list of terms.

Question: "{question}"

Concepts:'''

        try:
            response = generate_with_ollama(prompt, "qwen2.5:0.5b")
            if response:
                # Return original question + concepts for embedding context
                return f"{question} {response}"
        except Exception:
            pass
        
        return question

    def _reciprocal_rank_fusion(
        self, 
        embedding_scores: Dict[str, float], 
        bm25_scores: Dict[str, float], 
        k: int = 60
    ) -> Dict[str, float]:
        """
        Combine rankings using Reciprocal Rank Fusion (RRF).
        More robust than weighted average, no hyperparameter tuning needed.
        
        Args:
            embedding_scores: Semantic similarity scores
            bm25_scores: Lexical match scores
            k: Constant for RRF formula (default 60)
            
        Returns:
            Combined scores using RRF
        """
        # Rank by each method
        emb_ranked = sorted(embedding_scores.items(), key=lambda x: x[1], reverse=True)
        bm25_ranked = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        
        # RRF scoring: score = 1/(k + rank)
        rrf_scores: Dict[str, float] = {}
        
        for rank, (node_id, _) in enumerate(emb_ranked, 1):
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + 1.0 / (k + rank)
        
        for rank, (node_id, _) in enumerate(bm25_ranked, 1):
            rrf_scores[node_id] = rrf_scores.get(node_id, 0.0) + 1.0 / (k + rank)
        
        return rrf_scores

    def _build_table_text_map(self) -> Dict[str, str]:
        text_map = {}
        # Embed tables
        for schema in self.metadata.schemas.values():
            for table in schema.tables.values():
                node_id = f"table:{table.database}.{table.schema}.{table.name}"
                column_names = ", ".join(col.name for col in table.columns[:20])
                description = table.description or ""
                text = f"{table.name}. Description: {description}. Columns: {column_names}."
                text_map[node_id] = text
        
        # Embed concepts
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("kind") == "concept":
                text_map[node_id] = attrs.get("name", "")
                
        return text_map

    def _precompute_table_embeddings(self) -> Dict[str, List[float]]:
        if not self.refresh_cache:
            cached = self._load_cache()
            if cached:
                return cached

        embeddings: Dict[str, List[float]] = {}
        if not self.ollama_available:
            return embeddings

        for node_id, text in self.table_text_map.items():
            vector = embed_with_ollama(text, self.embedding_model)
            if vector:
                embeddings[node_id] = vector

        if embeddings:
            self._save_cache(embeddings)
        return embeddings

    def link(self, question: str, top_k: int = 10, alpha: float = 0.7, expand_query: bool = True) -> SchemaLinker.LinkerResult:
        """
        Link question to schema using the Corrected Workflow (Concept -> KG -> Structure).
        
        Args:
            question: User's natural language question
            top_k: Number of top candidates to return
            alpha: Weight for embedding score (1-alpha for BM25). Default 0.7.
            expand_query: Whether to use LLM for concept extraction. Default True.
        """
        if not self.table_embeddings:
            return super().link(question, top_k=top_k)

        print(f"\n[Step 1] Concept Extraction:")
        # Extract key concepts (nouns, entities) instead of hallucinating SQL
        search_query = self._extract_query_concepts(question) if expand_query else question
        print(f"  Original: {question}")
        print(f"  Concepts: {search_query}")

        query_embedding = embed_with_ollama(search_query, self.embedding_model)
        if not query_embedding:
            return super().link(question, top_k=top_k)

        # Track history for visualization
        history = []

        # 1. Embedding scores (Semantic)
        embedding_scores: Dict[str, float] = {}
        for node_id, vector in self.table_embeddings.items():
            embedding_scores[node_id] = cosine_similarity(query_embedding, vector)
        
        history.append({"step": "Semantic Retrieval", "scores": embedding_scores.copy()})
        
        print(f"\n[Step 2] Semantic Retrieval (Embeddings):")
        for node_id, score in sorted(embedding_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {node_id}: {score:.4f}")

        # 2. BM25 scores (Lexical)
        bm25_scores: Dict[str, float] = {}
        if self.bm25_index and self.bm25_node_ids:
            tokenized_query = self._tokenize(search_query)
            bm25_raw_scores = self.bm25_index.get_scores(tokenized_query)
            # Normalize BM25 scores to [0, 1]
            max_bm25 = max(bm25_raw_scores) if bm25_raw_scores.any() else 1.0
            for node_id, score in zip(self.bm25_node_ids, bm25_raw_scores):
                bm25_scores[node_id] = score / max_bm25 if max_bm25 > 0 else 0.0
        
        history.append({"step": "Lexical Retrieval", "scores": bm25_scores.copy()})

        print(f"\n[Step 3] Lexical Retrieval (BM25):")
        for node_id, score in sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {node_id}: {score:.4f}")

        # 3. Reciprocal Rank Fusion (RRF)
        # Filter out 0-score items to avoid ranking irrelevant noise
        emb_filtered = {k: v for k, v in embedding_scores.items() if v > 0.01} # minimal threshold for embeddings
        bm25_filtered = {k: v for k, v in bm25_scores.items() if v > 0}
        
        node_scores = self._reciprocal_rank_fusion(emb_filtered, bm25_filtered)
        history.append({"step": "Reciprocal Rank Fusion", "scores": node_scores.copy()})
        
        print(f"\n[Step 4] Reciprocal Rank Fusion (RRF):")
        for node_id, score in sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {node_id}: {score:.4f}")

        # 4. Propagate concept scores to tables
        table_scores: Dict[str, float] = {}
        for node_id, score in node_scores.items():
            if node_id.startswith("table:"):
                table_scores[node_id] = max(table_scores.get(node_id, 0.0), score)
            elif node_id.startswith("concept:"):
                # Propagate to related tables
                for neighbor in self.graph.predecessors(node_id):
                    if self.graph.nodes[neighbor].get("kind") == "table":
                        # Boost table score if it's related to a relevant concept
                        current = table_scores.get(neighbor, 0.0)
                        table_scores[neighbor] = current + (score * 0.5)
        
        history.append({"step": "Concept Propagation", "scores": table_scores.copy()})
        
        print(f"\n[Step 5] Concept Propagation:")
        for node_id, score in sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {node_id}: {score:.4f}")

        column_scores = self._score_columns(self._augment_tokens(self._tokenize(question)))
        table_scores = self._boost_tables_with_column_scores(table_scores, column_scores, weight=0.3)

        # 5. Graph Traversal & Reranking (FK-Aware Neighborhood Expansion)
        # This step expands the search to include joinable tables (1-hop neighbors)
        table_scores = self._rerank_with_graph_traversal(table_scores, decay_factor=0.3)
        
        history.append({"step": "Graph Traversal", "scores": table_scores.copy()})
        
        print(f"\n[Step 6] Graph Traversal Reranking (FK-Aware):")
        for node_id, score in sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {node_id}: {score:.4f}")
            
        column_scores = self._boost_columns_in_top_tables(column_scores, table_scores, boost_factor=0.2)

        # 6. Surgical Steiner Tree - Discover missing join tables
        # Refined: Use Column-Driven Terminals to avoid over-connection
        
        # 1. Get tables from Top 5 Columns (most precise signal)
        top_columns = sorted(column_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        col_terminals = set()
        for col_id, _ in top_columns:
            # col_id is "column:DB.SCHEMA.TABLE.COL"
            # Extract table ID: "table:DB.SCHEMA.TABLE"
            parts = col_id.split(".")
            if len(parts) >= 3:
                table_id = f"table:{'.'.join(parts[0].split(':')[1].split('.')[:3])}"
                # Fix: construct table ID properly from column ID
                # column:DB.SCHEMA.TABLE.COL -> table:DB.SCHEMA.TABLE
                # Actually, let's just use the graph structure if possible, or string parsing
                # String parsing is safer given the ID format
                table_part = ".".join(col_id.split(":")[1].split(".")[:-1])
                table_id = f"table:{table_part}"
                col_terminals.add(table_id)

        # 2. Get Top 3 Tables (direct semantic hits)
        top_tables_direct = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        table_terminals = {t[0] for t in top_tables_direct}
        
        # 3. Union and Limit
        # We prioritize column signals, then table signals
        all_terminals = list(col_terminals | table_terminals)
        # Limit to max 5 terminals to prevent giant graphs
        final_terminals = all_terminals[:5]
        
        print(f"  Steiner Terminals: {final_terminals}")

        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        sorted_tables = self._add_join_tables_via_steiner_tree(sorted_tables, top_k, terminals=final_terminals)
        
        # Create final scores map for history
        final_scores = {t[0]: t[1] for t in sorted_tables}
        history.append({"step": "Generated Schema", "scores": final_scores.copy()})
        
        print(f"\n[Step 7] Generated Schema (Steiner Tree):")
        print(f"  Final Top Tables: {[t[0] for t in sorted_tables]}")
        
        sorted_columns = sorted(column_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        from .schema_linker import LinkerResult  # avoid circular import at top

        return LinkerResult(
            question=question,
            candidate_tables=sorted_tables,
            candidate_columns=sorted_columns,
            score_history=history
        )

    def _add_join_tables_via_steiner_tree(
        self, 
        top_tables: List[Tuple[str, float]], 
        top_k: int,
        terminals: List[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Use Steiner tree to find missing join tables that connect top-ranked tables.
        
        Args:
            top_tables: List of (table_id, score) tuples
            top_k: Maximum number of tables to return
            terminals: Optional list of specific terminal nodes to connect. 
                       If None, uses top 3 tables.
            
        Returns:
            Updated list with join tables added
        """
        from .steiner_tree import steiner_tree_approximation
        
        # Use provided terminals or default to top 3 tables
        if not terminals:
            terminals = [t[0] for t in top_tables[:3]]
        
        # Filter terminals to ensure they exist in the graph
        valid_terminals = [t for t in terminals if t in self.graph]
        
        if len(valid_terminals) < 2:
            return top_tables
        
        # Find connecting nodes (join tables)
        join_tables = steiner_tree_approximation(self.graph, valid_terminals)
        
        # Add join tables with moderate score if not already in results
        existing_tables = {t[0] for t in top_tables}
        for table in join_tables:
            if table not in existing_tables:
                # Assign medium confidence score
                top_tables.append((table, 0.4))
        
        # Re-sort and limit to top_k
        return sorted(top_tables, key=lambda x: x[1], reverse=True)[:top_k]

    def _rerank_with_graph_traversal(self, table_scores: Dict[str, float], decay_factor: float = 0.3) -> Dict[str, float]:
        """
        Boost scores of tables related to high-scoring tables via foreign keys.
        
        Args:
            table_scores: Initial table scores
            decay_factor: Factor to decay propagated scores
            
        Returns:
            Updated table scores
        """
        boosted_scores = table_scores.copy()
        
        # Get top 3 tables
        top_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for table_id, score in top_tables:
            # Find related tables via HAS_COLUMN -> column -> foreign key relationships
            # For simplicity, boost tables in the same schema
            if table_id in self.graph:
                schema_node = list(self.graph.predecessors(table_id))[0] if list(self.graph.predecessors(table_id)) else None
                if schema_node:
                    for neighbor_table in self.graph.successors(schema_node):
                        if self.graph.nodes[neighbor_table].get("kind") == "table" and neighbor_table != table_id:
                            current = boosted_scores.get(neighbor_table, 0.0)
                            boosted_scores[neighbor_table] = current + (score * decay_factor)
        
        return boosted_scores

    def _boost_columns_in_top_tables(self, column_scores: Dict[str, float], table_scores: Dict[str, float], boost_factor: float = 0.2) -> Dict[str, float]:
        """
        Boost scores of columns that belong to high-scoring tables.
        
        Args:
            column_scores: Initial column scores
            table_scores: Table scores
            boost_factor: Factor to boost column scores
            
        Returns:
            Updated column scores
        """
        boosted_scores = column_scores.copy()
        
        # Get top 5 tables
        top_tables = set([table_id for table_id, _ in sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:5]])
        
        for column_id in column_scores.keys():
            # Find parent table
            if column_id in self.graph:
                parent_table = list(self.graph.predecessors(column_id))[0] if list(self.graph.predecessors(column_id)) else None
                if parent_table and parent_table in top_tables:
                    table_score = table_scores.get(parent_table, 0.0)
                    boosted_scores[column_id] += (table_score * boost_factor)
        
        return boosted_scores

    def _load_cache(self) -> Dict[str, List[float]]:
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return {node_id: vector for node_id, vector in data.items()}
        except (json.JSONDecodeError, OSError, ValueError):
            return {}

    def _save_cache(self, embeddings: Dict[str, List[float]]) -> None:
        try:
            with self.cache_path.open("w", encoding="utf-8") as f:
                json.dump(embeddings, f)
        except OSError:
            pass
