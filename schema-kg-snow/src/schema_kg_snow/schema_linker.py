"""Baseline schema linker for Spider2-Snow knowledge graph."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import networkx as nx

from .graph_builder import build_graph_from_metadata
from .metadata_loader import DatabaseMetadata


@dataclass
class LinkerResult:
    question: str
    candidate_tables: List[Tuple[str, float]]
    candidate_columns: List[Tuple[str, float]]
    score_history: List[Dict[str, Any]] = None  # List of {step: str, scores: Dict[str, float]}


class SchemaLinker:
    """Lightweight linker that scores tables/columns via lexical overlap."""

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.table_index = self._build_table_index()
        self.column_index = self._build_column_index()

    @classmethod
    def from_metadata(cls, metadata: DatabaseMetadata, **kwargs) -> "SchemaLinker":
        graph = build_graph_from_metadata(metadata, **kwargs)
        return cls(graph)

    def _build_table_index(self) -> Dict[str, Dict]:
        index: Dict[str, Dict] = {}
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("kind") != "table":
                continue
            name = attrs.get("name") or ""
            tokens = self._augment_tokens(self._tokenize(name))
            column_tokens = self._collect_table_column_tokens(node_id)
            index[node_id] = {
                "name_tokens": tokens,
                "description_tokens": self._augment_tokens(
                    self._tokenize(attrs.get("description", ""))
                ),
                "column_tokens": column_tokens,
            }
        return index

    def _build_column_index(self) -> Dict[str, Dict]:
        index: Dict[str, Dict] = {}
        for node_id, attrs in self.graph.nodes(data=True):
            if attrs.get("kind") != "column":
                continue
            name = attrs.get("name") or ""
            tokens = self._augment_tokens(self._tokenize(name))
            description_tokens = self._augment_tokens(self._tokenize(attrs.get("description", "")))
            
            # Include sample values in the index
            sample_values = attrs.get("sample_values", [])
            sample_tokens = []
            if sample_values:
                for val in sample_values:
                    if isinstance(val, str):
                        sample_tokens.extend(self._tokenize(val))
                sample_tokens = self._augment_tokens(sample_tokens)
            
            table_node = self._get_parent_table(node_id)
            table_tokens = self.table_index.get(table_node, {}).get("name_tokens", [])
            index[node_id] = {
                "name_tokens": tokens,
                "description_tokens": description_tokens,
                "sample_tokens": sample_tokens,
                "table_node": table_node,
                "table_tokens": table_tokens,
            }
        return index

    def _get_parent_table(self, column_node: str) -> str | None:
        predecessors = list(self.graph.predecessors(column_node))
        for pred in predecessors:
            if self.graph.nodes[pred].get("kind") == "table":
                return pred
        return None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = (text or "").lower()
        raw_tokens = [tok for tok in re.split(r"[^a-z0-9]+", text) if tok]
        return [SchemaLinker._normalize_token(tok) for tok in raw_tokens]

    def _augment_tokens(self, tokens: List[str]) -> List[str]:
        expanded = set(tokens)
        # Synonyms removed in favor of GraphRAG
        expanded.update(self._generate_ngrams(tokens, 2))
        expanded.update(self._generate_ngrams(tokens, 3))
        return list(expanded)

    def _generate_ngrams(self, tokens: List[str], n: int) -> List[str]:
        if len(tokens) < n:
            return []
        grams = []
        for i in range(len(tokens) - n + 1):
            grams.append("_".join(tokens[i : i + n]))
        return grams

    def _collect_table_column_tokens(self, table_node: str) -> List[str]:
        tokens: List[str] = []
        for neighbor in self.graph.successors(table_node):
            neighbor_attrs = self.graph.nodes[neighbor]
            if neighbor_attrs.get("kind") != "column":
                continue
            col_name = neighbor_attrs.get("name") or ""
            tokens.extend(self._augment_tokens(self._tokenize(col_name)))
            col_desc = neighbor_attrs.get("description", "")
            if col_desc:
                tokens.extend(self._augment_tokens(self._tokenize(col_desc)))
            
            # Add sample values from columns to table tokens
            sample_values = neighbor_attrs.get("sample_values", [])
            if sample_values:
                for val in sample_values:
                    if isinstance(val, str):
                        tokens.extend(self._augment_tokens(self._tokenize(val)))
        return list(set(tokens))

    @staticmethod
    def _normalize_token(token: str) -> str:
        if token.endswith("ies") and len(token) > 3:
            return token[:-3] + "y"
        if token.endswith("s") and len(token) > 3:
            return token[:-1]
        return token

    def link(self, question: str, top_k: int = 10) -> LinkerResult:
        question_tokens = self._augment_tokens(self._tokenize(question))
        table_scores = self._score_tables(question_tokens)
        column_scores = self._score_columns(question_tokens)

        table_scores = self._boost_tables_with_column_scores(table_scores, column_scores)

        sorted_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        sorted_columns = sorted(column_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        return LinkerResult(
            question=question,
            candidate_tables=sorted_tables,
            candidate_columns=sorted_columns,
        )

    def _score_tables(self, question_tokens: List[str]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        question_set = set(question_tokens)

        for node_id, tokens in self.table_index.items():
            name_tokens = tokens["name_tokens"]
            desc_tokens = tokens["description_tokens"]
            column_tokens = tokens.get("column_tokens", [])

            score = self._overlap_score(question_set, name_tokens)
            score += 0.5 * self._overlap_score(question_set, desc_tokens)
            score += 0.7 * self._overlap_score(question_set, column_tokens)
            if score > 0:
                scores[node_id] = score
        return scores

    def _score_columns(self, question_tokens: List[str]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        question_set = set(question_tokens)

        for node_id, tokens in self.column_index.items():
            name_tokens = tokens["name_tokens"]
            table_tokens = tokens.get("table_tokens", [])
            desc_tokens = tokens.get("description_tokens", [])
            score = self._overlap_score(question_set, name_tokens)
            score += 0.3 * self._overlap_score(question_set, table_tokens)
            score += 0.4 * self._overlap_score(question_set, desc_tokens)
            
            # Add score for sample values
            sample_tokens = tokens.get("sample_tokens", [])
            if sample_tokens:
                score += 0.6 * self._overlap_score(question_set, sample_tokens)
                
            if score > 0:
                scores[node_id] = score
        return scores

    def _boost_tables_with_column_scores(
        self,
        table_scores: Dict[str, float],
        column_scores: Dict[str, float],
        weight: float = 0.8,
    ) -> Dict[str, float]:
        boosted = dict(table_scores)
        for column_node, score in column_scores.items():
            table_node = self.column_index.get(column_node, {}).get("table_node")
            if not table_node:
                continue
            boosted[table_node] = boosted.get(table_node, 0.0) + score * weight
        return boosted

    @staticmethod
    def _overlap_score(question_tokens: set[str], target_tokens: List[str]) -> float:
        if not target_tokens:
            return 0.0
        target_set = set(target_tokens)
        overlap = question_tokens.intersection(target_set)
        if not overlap:
            return 0.0
        return len(overlap) / len(target_set)
