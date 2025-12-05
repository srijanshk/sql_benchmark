"""Graph RAG schema linker powered by Ollama embeddings."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from rank_bm25 import BM25Okapi

from .graph_builder import build_graph_from_metadata
from .metadata_loader import DatabaseMetadata
from .schema_linker import SchemaLinker
from .ollama_utils import embed_with_ollama, check_ollama_connection

logger = logging.getLogger(__name__)


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
        use_column_embeddings: bool = False,  # Disabled by default for speed
    ):
        graph = build_graph_from_metadata(metadata, concept_model=concept_model)
        super().__init__(graph)
        self.metadata = metadata
        self.embedding_model = embedding_model
        self.cache_dir = Path(cache_dir or "schema-kg-snow/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / f"{self.metadata.name}_embeddings.json"
        self.refresh_cache = refresh_cache
        self.use_column_embeddings = use_column_embeddings
        self.ollama_available = check_ollama_connection()
        self.table_text_map = self._build_table_text_map()
        self.column_text_map = self._build_column_text_map() if use_column_embeddings else {}
        self.table_embeddings = self._precompute_table_embeddings()
        self.column_embeddings = self._precompute_column_embeddings() if use_column_embeddings else {}
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
        Extract key concepts from question with ANTI-HALLUCINATION FILTER.
        Validates concepts against actual schema vocabulary.
        
        Args:
            question: Original user question
            
        Returns:
            String containing validated concepts appended to question
        """
        from .ollama_utils import generate_with_ollama
        from .extraction.concept_validator import filter_and_rank_concepts
        
        prompt = f'''Extract the key business concepts, entities, and values from the following question.
Focus on nouns, metrics, and specific filters.
Do NOT generate SQL. Do NOT guess table names.
Output ONLY a comma-separated list of terms.

Question: "{question}"

Concepts:'''

        try:
            response = generate_with_ollama(prompt, "qwen2.5:0.5b")
            if response:
                # Parse LLM response into concept list
                raw_concepts = [c.strip() for c in response.split(',') if c.strip()]
                
                # CRITICAL: Validate against schema vocabulary
                validated_concepts = filter_and_rank_concepts(
                    raw_concepts, 
                    self.metadata, 
                    top_k=5
                )
                
                if validated_concepts:
                    # Return question + validated concepts
                    concepts_str = ", ".join(validated_concepts)
                    return f"{question} {concepts_str}"
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
    
    def _build_column_text_map(self) -> Dict[str, str]:
        """Build text representations for columns."""
        text_map = {}
        
        for schema in self.metadata.schemas.values():
            for table in schema.tables.values():
                for col in table.columns:
                    node_id = f"column:{table.database}.{table.schema}.{table.name}.{col.name}"
                    
                    # Build rich column text
                    parts = [
                        f"Column: {col.name}",
                        f"Type: {col.data_type}",
                        f"Table: {table.name}"
                    ]
                    
                    if col.description:
                        parts.append(f"Description: {col.description}")
                    
                    if col.sample_values:
                        sample_str = ", ".join(str(v)[:50] for v in col.sample_values[:3] if v is not None)
                        if sample_str:
                            parts.append(f"Examples: {sample_str}")
                    
                    text_map[node_id] = ". ".join(parts)
        
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
    
    def _precompute_column_embeddings(self) -> Dict[str, List[float]]:
        """Precompute embeddings for all columns."""
        cache_path = self.cache_dir / f"{self.metadata.name}_column_embeddings.json"
        
        # Try to load from cache
        if not self.refresh_cache and cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                    logger.info(f"Loaded {len(cached)} column embeddings from cache")
                    return cached
            except Exception as e:
                logger.warning(f"Failed to load column embedding cache: {e}")
        
        embeddings: Dict[str, List[float]] = {}
        if not self.ollama_available:
            return embeddings
        
        logger.info(f"Computing embeddings for {len(self.column_text_map)} columns...")
        
        for i, (node_id, text) in enumerate(self.column_text_map.items(), 1):
            vector = embed_with_ollama(text, self.embedding_model)
            if vector:
                embeddings[node_id] = vector
            
            if i % 50 == 0:
                logger.info(f"  Embedded {i}/{len(self.column_text_map)} columns")
        
        # Save to cache
        if embeddings:
            try:
                with open(cache_path, 'w') as f:
                    json.dump(embeddings, f)
                logger.info(f"Saved {len(embeddings)} column embeddings to cache")
            except Exception as e:
                logger.warning(f"Failed to save column embedding cache: {e}")
        
        return embeddings

    def link(self, question: str, top_k: int = 10, alpha: float = 0.7, expand_query: bool = True) -> SchemaLinker.LinkerResult:
        """
        Link question to schema using the REFINED Workflow with Anti-Hallucination Filters.
        
        Pipeline:
        1. Extract & validate concepts (no hallucinations)
        2. Extract constraints (dates, values, filters)
        3. Hybrid retrieval (semantic + lexical)
        4. FK-aware graph traversal
        5. Column role classification
        6. Join path discovery
        
        Args:
            question: User's natural language question
            top_k: Number of top candidates to return
            alpha: Weight for embedding score (1-alpha for BM25). Default 0.7.
            expand_query: Whether to use LLM for concept extraction. Default True.
        """
        if not self.table_embeddings:
            return super().link(question, top_k=top_k)

        # Step 1: Extract and validate concepts (ANTI-HALLUCINATION)
        print(f"\n[Step 1] Concept Extraction (Validated):")
        search_query = self._extract_query_concepts(question) if expand_query else question
        
        # Extract validated concepts for output
        from .extraction.concept_validator import filter_and_rank_concepts
        from .ollama_utils import generate_with_ollama
        
        validated_concepts = []
        if expand_query:
            response = generate_with_ollama(
                f'Extract key business concepts from: "{question}". Output comma-separated list only.',
                "qwen2.5:0.5b"
            )
            if response:
                raw_concepts = [c.strip() for c in response.split(',') if c.strip()]
                validated_concepts = filter_and_rank_concepts(raw_concepts, self.metadata, top_k=5)
        
        print(f"  Validated concepts: {validated_concepts}")
        
        # Step 2: Extract constraints (NEW!)
        print(f"\n[Step 2] Constraint Extraction:")
        from .extraction.constraint_extractor import extract_constraints
        
        constraints = extract_constraints(question, use_llm=False)  # Regex only for now
        print(f"  Extracted constraints: {constraints}")

        query_embedding = embed_with_ollama(search_query, self.embedding_model)
        if not query_embedding:
            return super().link(question, top_k=top_k)

        # Track history for visualization
        history = []

        # Step 3: Embedding scores (Semantic)
        print(f"\n[Step 3] Semantic Retrieval (Embeddings):")
        embedding_scores: Dict[str, float] = {}
        for node_id, vector in self.table_embeddings.items():
            embedding_scores[node_id] = cosine_similarity(query_embedding, vector)
        
        history.append({"step": "Semantic Retrieval", "scores": embedding_scores.copy()})
        
        for node_id, score in sorted(embedding_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {node_id}: {score:.4f}")

        # Step 4: BM25 scores (Lexical)
        print(f"\n[Step 4] Lexical Retrieval (BM25):")
        bm25_scores: Dict[str, float] = {}
        if self.bm25_index and self.bm25_node_ids:
            tokenized_query = self._tokenize(search_query)
            bm25_raw_scores = self.bm25_index.get_scores(tokenized_query)
            # Normalize BM25 scores to [0, 1]
            max_bm25 = max(bm25_raw_scores) if bm25_raw_scores.any() else 1.0
            for node_id, score in zip(self.bm25_node_ids, bm25_raw_scores):
                bm25_scores[node_id] = score / max_bm25 if max_bm25 > 0 else 0.0
        
        history.append({"step": "Lexical Retrieval", "scores": bm25_scores.copy()})

        for node_id, score in sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {node_id}: {score:.4f}")

        # Step 5: Reciprocal Rank Fusion (RRF)
        print(f"\n[Step 5] Reciprocal Rank Fusion (RRF):")
        # Filter out 0-score items to avoid ranking irrelevant noise
        emb_filtered = {k: v for k, v in embedding_scores.items() if v > 0.01}
        bm25_filtered = {k: v for k, v in bm25_scores.items() if v > 0}
        
        node_scores = self._reciprocal_rank_fusion(emb_filtered, bm25_filtered)
        history.append({"step": "Reciprocal Rank Fusion", "scores": node_scores.copy()})
        
        for node_id, score in sorted(node_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {node_id}: {score:.4f}")

        # Step 6: Concept Propagation (with validated concepts only)
        print(f"\n[Step 6] Concept Propagation:")
        table_scores: Dict[str, float] = {}
        for node_id, score in node_scores.items():
            if node_id.startswith("table:"):
                table_scores[node_id] = max(table_scores.get(node_id, 0.0), score)
            elif node_id.startswith("concept:"):
                # Only propagate if concept was validated
                concept_name = node_id.split(":")[-1]
                if concept_name in validated_concepts:
                    # Propagate to related tables
                    for neighbor in self.graph.predecessors(node_id):
                        if self.graph.nodes[neighbor].get("kind") == "table":
                            current = table_scores.get(neighbor, 0.0)
                            table_scores[neighbor] = current + (score * 0.5)
        
        history.append({"step": "Concept Propagation", "scores": table_scores.copy()})
        
        for node_id, score in sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {node_id}: {score:.4f}")

        # Column scoring (OPTIONAL - only if enabled)
        if self.use_column_embeddings and self.column_embeddings:
            column_scores = self._score_columns(self._augment_tokens(self._tokenize(question)))
            table_scores = self._boost_tables_with_column_scores(table_scores, column_scores, weight=0.3)
        else:
            # Fallback: use lexical column matching
            column_scores = {}

        # Step 7: FK-AWARE Graph Traversal (REFINED!)
        # Expands using FOREIGN_KEY edges (Priority 1), not just schema proximity
        print(f"\n[Step 7] FK-Aware Graph Traversal:")
        table_scores = self._rerank_with_graph_traversal(table_scores, decay_factor=0.3)
        
        history.append({"step": "FK-Aware Graph Traversal", "scores": table_scores.copy()})
        
        for node_id, score in sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"  {node_id}: {score:.4f}")
            
        # Column boost (only if enabled)
        if self.use_column_embeddings and column_scores:
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
        
        print(f"\n[Step 8] Final Schema Generation:")
        print(f"  Top Tables: {[self._extract_table_name(t[0]) for t in sorted_tables[:5]]}")
        
        # Get column candidates (use column embeddings if available)
        if self.use_column_embeddings and column_scores:
            sorted_columns = sorted(column_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        else:
            # Fallback: lexical column matching from question tokens
            sorted_columns = self._get_columns_lexical(question, sorted_tables, top_k)

        # Import new result types
        from .schema_linker import LinkerResult, TableCandidate, ColumnCandidate, JoinPath
        
        # Build table candidates with explanations
        table_candidates = []
        for node_id, score in sorted_tables:
            table_name = self._extract_table_name(node_id)
            why_reasons = self._explain_table_selection(node_id, validated_concepts, constraints, embedding_scores, bm25_scores)
            table_candidates.append(TableCandidate(
                table=table_name,
                score=score,
                why=why_reasons
            ))
        
        # Build column candidates with role classification
        column_candidates = []
        for node_id, score in sorted_columns:
            column_name = self._extract_column_name(node_id)
            role = self._classify_column_role(node_id, constraints, question)
            why_reasons = self._explain_column_selection(node_id, constraints, role)
            column_candidates.append(ColumnCandidate(
                column=column_name,
                score=score,
                role=role,
                why=why_reasons
            ))
        
        # Build join paths from FK edges
        join_paths = self._discover_join_paths(sorted_tables[:5])
        
        # Convert constraints to output format
        constraints_output = [
            {
                "type": c.type.value,
                "operator": c.operator,
                "value": str(c.value),
                "column_hint": c.column_hint
            }
            for c in constraints
        ]

        return LinkerResult(
            question=question,
            concepts=validated_concepts,
            constraints=constraints_output,
            candidate_tables=table_candidates,
            candidate_columns=column_candidates,
            join_paths=join_paths,
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
        FK-AWARE Graph Traversal: Boost scores of tables related via actual Foreign Keys.
        
        Priority system (following Database Physics):
        1. FOREIGN_KEY edges: Weight 1.0 (actual FK relationships)
        2. SIMILAR_NAME edges: Weight 0.5 (heuristic name matching)
        3. SAME_SCHEMA: Weight 0.1 (weak signal, schema proximity)
        
        Args:
            table_scores: Initial table scores
            decay_factor: Base factor to decay propagated scores
            
        Returns:
            Updated table scores with FK-boosted neighbors
        """
        boosted_scores = table_scores.copy()
        
        # Get top 3 tables as "seed" tables for expansion
        top_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for table_id, score in top_tables:
            if table_id not in self.graph:
                continue
            
            # Priority 1: Foreign Key neighbors (HIGHEST priority)
            fk_neighbors = self._get_fk_neighbors(table_id)
            for neighbor, direction in fk_neighbors:
                current = boosted_scores.get(neighbor, 0.0)
                # Full decay factor for FK (strong signal)
                boosted_scores[neighbor] = current + (score * decay_factor * 1.0)
            
            # Priority 2: Similar name neighbors (MEDIUM priority)
            similar_neighbors = self._get_similar_name_neighbors(table_id)
            for neighbor in similar_neighbors:
                if neighbor not in fk_neighbors:  # Don't double-boost
                    current = boosted_scores.get(neighbor, 0.0)
                    # Half decay for heuristic matches
                    boosted_scores[neighbor] = current + (score * decay_factor * 0.5)
            
            # Priority 3: Same schema neighbors (LOWEST priority)
            schema_neighbors = self._get_schema_neighbors(table_id)
            for neighbor in schema_neighbors:
                if neighbor not in fk_neighbors and neighbor not in similar_neighbors:
                    current = boosted_scores.get(neighbor, 0.0)
                    # Very weak boost for schema proximity
                    boosted_scores[neighbor] = current + (score * decay_factor * 0.1)
        
        return boosted_scores
    
    def _get_fk_neighbors(self, table_id: str) -> List[Tuple[str, str]]:
        """
        Get tables connected via FOREIGN_KEY edges.
        
        Returns:
            List of (neighbor_table_id, direction) where direction is 'outgoing' or 'incoming'
        """
        neighbors = []
        
        # Outgoing FK: This table references another table
        for neighbor in self.graph.successors(table_id):
            edge_data = self.graph.get_edge_data(table_id, neighbor)
            if edge_data and edge_data.get('rel') == 'FOREIGN_KEY':
                neighbors.append((neighbor, 'outgoing'))
        
        # Incoming FK: Another table references this table
        for neighbor in self.graph.predecessors(table_id):
            edge_data = self.graph.get_edge_data(neighbor, table_id)
            if edge_data and edge_data.get('rel') == 'FOREIGN_KEY':
                neighbors.append((neighbor, 'incoming'))
        
        return neighbors
    
    def _get_similar_name_neighbors(self, table_id: str) -> List[str]:
        """
        Get tables connected via SIMILAR_NAME edges (heuristic joins).
        """
        neighbors = []
        
        for neighbor in self.graph.successors(table_id):
            edge_data = self.graph.get_edge_data(table_id, neighbor)
            if edge_data and edge_data.get('rel') == 'SIMILAR_NAME':
                neighbors.append(neighbor)
        
        for neighbor in self.graph.predecessors(table_id):
            edge_data = self.graph.get_edge_data(neighbor, table_id)
            if edge_data and edge_data.get('rel') == 'SIMILAR_NAME':
                neighbors.append(neighbor)
        
        return neighbors
    
    def _get_schema_neighbors(self, table_id: str) -> List[str]:
        """
        Get all tables in the same schema (fallback, weakest signal).
        """
        neighbors = []
        
        # Find schema node
        schema_node = None
        for pred in self.graph.predecessors(table_id):
            if self.graph.nodes[pred].get("kind") == "schema":
                schema_node = pred
                break
        
        if schema_node:
            # Get all tables in this schema
            for neighbor in self.graph.successors(schema_node):
                if (self.graph.nodes[neighbor].get("kind") == "table" and 
                    neighbor != table_id):
                    neighbors.append(neighbor)
        
        return neighbors

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

    def _extract_table_name(self, node_id: str) -> str:
        """Extract DB.SCHEMA.TABLE from node_id."""
        if ":" in node_id:
            return node_id.split(":", 1)[1]
        return node_id
    
    def _extract_column_name(self, node_id: str) -> str:
        """Extract DB.SCHEMA.TABLE.COLUMN from node_id."""
        if ":" in node_id:
            return node_id.split(":", 1)[1]
        return node_id
    
    def _explain_table_selection(
        self,
        table_id: str,
        concepts: List[str],
        constraints: List,
        semantic_scores: Dict[str, float],
        lexical_scores: Dict[str, float]
    ) -> List[str]:
        """Generate human-readable explanations for why a table was selected."""
        reasons = []
        
        # Check semantic match
        if table_id in semantic_scores and semantic_scores[table_id] > 0.5:
            reasons.append(f"High semantic similarity ({semantic_scores[table_id]:.2f})")
        
        # Check lexical match
        if table_id in lexical_scores and lexical_scores[table_id] > 0.3:
            reasons.append("Strong lexical match")
        
        # Check concept connections
        table_concepts = []
        for neighbor in self.graph.successors(table_id):
            if self.graph.nodes[neighbor].get("kind") == "concept":
                concept_name = self.graph.nodes[neighbor].get("name", "").lower()
                if concept_name in concepts:
                    table_concepts.append(concept_name)
        
        if table_concepts:
            reasons.append(f"Related to concepts: {', '.join(table_concepts)}")
        
        # Check FK connections
        fk_neighbors = self._get_fk_neighbors(table_id)
        if fk_neighbors:
            reasons.append(f"Connected via {len(fk_neighbors)} foreign keys")
        
        if not reasons:
            reasons.append("Candidate from schema linking")
        
        return reasons
    
    def _explain_column_selection(
        self,
        column_id: str,
        constraints: List,
        role: str
    ) -> List[str]:
        """Generate explanations for column selection."""
        reasons = []
        
        if role:
            reasons.append(f"Role: {role}")
        
        # Check if column matches constraints
        column_name = column_id.split(".")[-1].lower()
        for constraint in constraints:
            if constraint.column_hint and constraint.column_hint.lower() in column_name:
                reasons.append(f"Matches constraint: {constraint.type.value}")
        
        # Check column attributes
        if column_id in self.graph:
            attrs = self.graph.nodes[column_id]
            data_type = attrs.get("data_type", "")
            if "date" in data_type.lower() or "time" in data_type.lower():
                reasons.append("Temporal column")
            elif "int" in data_type.lower() or "numeric" in data_type.lower():
                reasons.append("Numeric column")
        
        if not reasons:
            reasons.append("Lexical match with question")
        
        return reasons
    
    def _classify_column_role(self, column_id: str, constraints: List, question: str) -> str:
        """Classify column role: join_key, filter, measure, or dimension."""
        if column_id not in self.graph:
            return None
        
        attrs = self.graph.nodes[column_id]
        column_name = attrs.get("name", "").lower()
        data_type = attrs.get("data_type", "").lower()
        
        # 1. Join key detection
        table_id = list(self.graph.predecessors(column_id))[0] if list(self.graph.predecessors(column_id)) else None
        if table_id:
            fk_neighbors = self._get_fk_neighbors(table_id)
            if fk_neighbors or "_id" in column_name or column_name.endswith("id"):
                return "join_key"
        
        # 2. Filter column (matches constraints)
        for constraint in constraints:
            if constraint.column_hint and constraint.column_hint.lower() in column_name:
                return "filter"
        
        # 3. Measure column (numeric aggregations)
        if any(kw in question.lower() for kw in ["sum", "total", "count", "average", "max", "min", "top"]):
            if any(t in data_type for t in ["int", "numeric", "decimal", "float"]):
                return "measure"
        
        # 4. Dimension column (GROUP BY)
        if any(kw in question.lower() for kw in ["by", "per", "each", "group"]):
            if "varchar" in data_type or "string" in data_type:
                return "dimension"
        
        return None
    
    def _discover_join_paths(self, top_tables: List[Tuple[str, float]]) -> List:
        """Discover FK-based join paths between top tables."""
        from .schema_linker import JoinPath
        
        join_paths = []
        table_ids = [t[0] for t in top_tables]
        
        # Find direct FK connections
        for i, table1 in enumerate(table_ids):
            for table2 in table_ids[i+1:]:
                # Check FK edge
                edge_data = self.graph.get_edge_data(table1, table2)
                if edge_data and edge_data.get('rel') == 'FOREIGN_KEY':
                    join_paths.append(JoinPath(
                        tables=[self._extract_table_name(table1), self._extract_table_name(table2)],
                        joins=[{"left": self._extract_table_name(table1), "right": self._extract_table_name(table2)}],
                        score=0.9
                    ))
        
        return join_paths
    
    def _score_columns(self, query_tokens: List[str]) -> Dict[str, float]:
        """
        Score columns using semantic embeddings for precision improvement.
        
        Args:
            query_tokens: Tokenized question
            
        Returns:
            Dict mapping column_id -> score
        """
        column_scores = {}
        
        if not self.column_embeddings:
            # Fallback to lexical matching
            query_text = " ".join(query_tokens).lower()
            for col_id, col_text in self.column_text_map.items():
                # Simple token overlap
                col_tokens = set(self._tokenize(col_text.lower()))
                query_token_set = set(query_tokens)
                overlap = len(col_tokens & query_token_set)
                if overlap > 0:
                    column_scores[col_id] = overlap / len(query_token_set)
            return column_scores
        
        # Semantic scoring with embeddings
        query_text = " ".join(query_tokens)
        query_embedding = embed_with_ollama(query_text, self.embedding_model)
        
        if not query_embedding:
            return column_scores
        
        for col_id, col_embedding in self.column_embeddings.items():
            score = cosine_similarity(query_embedding, col_embedding)
            if score > 0.3:  # Threshold to filter noise
                column_scores[col_id] = score
        
        return column_scores
    
    def _boost_tables_with_column_scores(
        self,
        table_scores: Dict[str, float],
        column_scores: Dict[str, float],
        weight: float = 0.3
    ) -> Dict[str, float]:
        """
        Boost table scores based on their column relevance (Precision Improvement).
        
        If a table has many high-scoring columns, it's more likely to be relevant.
        This helps filter out tables that only match at the table level but have
        no relevant columns.
        
        Args:
            table_scores: Current table scores
            column_scores: Column relevance scores
            weight: How much to weight column signals (0-1)
            
        Returns:
            Boosted table scores
        """
        boosted_scores = table_scores.copy()
        
        # Aggregate column scores per table
        table_column_agg = defaultdict(list)
        for col_id, score in column_scores.items():
            # Extract table from column:DB.SCHEMA.TABLE.COL
            parts = col_id.split(":")
            if len(parts) == 2:
                col_path = parts[1]
                table_path = ".".join(col_path.split(".")[:-1])
                table_id = f"table:{table_path}"
                table_column_agg[table_id].append(score)
        
        # Boost tables based on top-3 column scores
        for table_id, col_scores_list in table_column_agg.items():
            if table_id in boosted_scores:
                # Use top-3 columns' average score
                top_col_scores = sorted(col_scores_list, reverse=True)[:3]
                avg_col_score = sum(top_col_scores) / len(top_col_scores) if top_col_scores else 0.0
                
                # Boost table score
                current_score = boosted_scores[table_id]
                boosted_scores[table_id] = current_score + (avg_col_score * weight)
        
        return boosted_scores
    
    def _boost_columns_in_top_tables(
        self,
        column_scores: Dict[str, float],
        table_scores: Dict[str, float],
        boost_factor: float = 0.2
    ) -> Dict[str, float]:
        """
        Boost columns that belong to top-ranked tables.
        
        This creates a feedback loop: good tables -> boost their columns.
        
        Args:
            column_scores: Current column scores
            table_scores: Table relevance scores
            boost_factor: Boost multiplier
            
        Returns:
            Boosted column scores
        """
        boosted_cols = column_scores.copy()
        
        # Get top-5 tables
        top_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        top_table_ids = {t[0] for t in top_tables}
        
        # Boost columns in these tables
        for col_id in list(boosted_cols.keys()):
            parts = col_id.split(":")
            if len(parts) == 2:
                col_path = parts[1]
                table_path = ".".join(col_path.split(".")[:-1])
                table_id = f"table:{table_path}"
                
                if table_id in top_table_ids:
                    boosted_cols[col_id] *= (1 + boost_factor)
        
        return boosted_cols
    
    def _get_columns_lexical(self, question: str, top_tables: List[Tuple[str, float]], top_k: int) -> List[Tuple[str, float]]:
        """Fallback lexical column matching when embeddings disabled."""
        column_scores = {}
        query_tokens = set(self._tokenize(question.lower()))
        
        # Get columns from top tables
        for table_id, _ in top_tables[:5]:
            table_name = self._extract_table_name(table_id)
            parts = table_name.split(".")
            if len(parts) == 3:
                db, schema, tbl = parts
                if db in self.metadata.name and schema in self.metadata.schemas:
                    schema_obj = self.metadata.schemas[schema]
                    if tbl in schema_obj.tables:
                        table_obj = schema_obj.tables[tbl]
                        for col in table_obj.columns:
                            col_id = f"column:{db}.{schema}.{tbl}.{col.name}"
                            col_tokens = set(self._tokenize(col.name.lower()))
                            overlap = len(col_tokens & query_tokens)
                            if overlap > 0:
                                column_scores[col_id] = overlap / len(query_tokens)
        
        return sorted(column_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
