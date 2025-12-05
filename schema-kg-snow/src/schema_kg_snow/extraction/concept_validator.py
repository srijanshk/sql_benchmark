"""Validate and filter LLM-generated concepts against actual schema vocabulary."""

from __future__ import annotations

from typing import List, Set
import re

from ..metadata_loader import DatabaseMetadata


# Blacklist of generic terms that LLMs commonly hallucinate
CONCEPT_BLACKLIST = {
    "etc", "and_more", "description", "table", "column", "database", "schema",
    "type", "name", "value", "data", "information", "record", "entry", "item",
    "field", "attribute", "property", "metadata", "such_as", "and", "or",
    "such_as_traffic_data", "such_as_page_views", "bounce_rate", "traffic_sources",
    "the", "a", "an", "of", "to", "in", "for", "on", "at", "from", "by",
}


def build_schema_vocabulary(metadata: DatabaseMetadata) -> Set[str]:
    """
    Build a vocabulary of valid terms from the database schema.
    This includes table names, column names, and meaningful tokens from descriptions.
    
    Args:
        metadata: Database metadata
        
    Returns:
        Set of normalized vocabulary terms
    """
    vocabulary = set()
    
    for schema in metadata.schemas.values():
        # Add schema name tokens
        vocabulary.update(_tokenize(schema.name))
        
        for table in schema.tables.values():
            # Add table name tokens
            vocabulary.update(_tokenize(table.name))
            
            # Add description tokens (filtered)
            if table.description:
                desc_tokens = _tokenize(table.description)
                # Keep only meaningful tokens (length > 3, not in blacklist)
                meaningful = {t for t in desc_tokens 
                             if len(t) > 3 and t not in CONCEPT_BLACKLIST}
                vocabulary.update(meaningful)
            
            # Add column names
            for column in table.columns:
                vocabulary.update(_tokenize(column.name))
                
                # Add column description tokens
                if column.description:
                    desc_tokens = _tokenize(column.description)
                    meaningful = {t for t in desc_tokens 
                                 if len(t) > 3 and t not in CONCEPT_BLACKLIST}
                    vocabulary.update(meaningful)
                
                # Add sample values (categorical only)
                if column.sample_values:
                    for val in column.sample_values[:5]:  # Limit to first 5
                        if isinstance(val, str) and len(val) < 50:  # Avoid long text
                            vocabulary.update(_tokenize(val))
    
    return vocabulary


def validate_concepts(
    concepts: List[str], 
    schema_vocabulary: Set[str],
    min_confidence: float = 0.5
) -> List[tuple[str, float]]:
    """
    Validate and score concepts against schema vocabulary.
    
    Args:
        concepts: List of concepts extracted by LLM
        schema_vocabulary: Valid terms from database schema
        min_confidence: Minimum confidence score to keep concept
        
    Returns:
        List of (concept, confidence_score) tuples, sorted by confidence
    """
    validated = []
    
    for concept in concepts:
        concept_clean = concept.lower().strip()
        
        # Rule 1: Reject blacklisted terms
        if concept_clean in CONCEPT_BLACKLIST:
            continue
        
        # Rule 2: Reject too short or too long
        if len(concept_clean) < 3 or len(concept_clean) > 50:
            continue
        
        # Rule 3: Calculate confidence based on schema match
        confidence = _calculate_confidence(concept_clean, schema_vocabulary)
        
        if confidence >= min_confidence:
            validated.append((concept_clean, confidence))
    
    # Sort by confidence (highest first)
    validated.sort(key=lambda x: x[1], reverse=True)
    
    return validated


def _calculate_confidence(concept: str, vocabulary: Set[str]) -> float:
    """
    Calculate confidence score for a concept based on schema vocabulary.
    
    Scoring logic:
    - 1.0: Exact match in vocabulary
    - 0.9: Substring match (concept is part of a vocab term)
    - 0.7: Partial token match (some tokens overlap)
    - 0.3: Fuzzy match (edit distance)
    - 0.0: No match
    """
    concept_norm = concept.lower()
    
    # Exact match
    if concept_norm in vocabulary:
        return 1.0
    
    # Substring match (concept appears in a vocabulary term)
    for vocab_term in vocabulary:
        if concept_norm in vocab_term or vocab_term in concept_norm:
            return 0.9
    
    # Token overlap match
    concept_tokens = set(_tokenize(concept))
    for vocab_term in vocabulary:
        vocab_tokens = set(_tokenize(vocab_term))
        overlap = concept_tokens & vocab_tokens
        if overlap:
            overlap_ratio = len(overlap) / len(concept_tokens)
            if overlap_ratio >= 0.5:
                return 0.7
    
    # Fuzzy match using edit distance
    for vocab_term in vocabulary:
        if _fuzzy_match(concept_norm, vocab_term):
            return 0.3
    
    return 0.0


def _tokenize(text: str) -> Set[str]:
    """Tokenize text into normalized words."""
    if not text:
        return set()
    
    text = text.lower()
    # Split on non-alphanumeric characters
    tokens = re.split(r'[^a-z0-9]+', text)
    # Filter empty and blacklisted
    tokens = {t for t in tokens if t and t not in CONCEPT_BLACKLIST}
    
    return tokens


def _fuzzy_match(s1: str, s2: str, threshold: int = 2) -> bool:
    """
    Check if two strings are similar using Levenshtein distance.
    
    Args:
        s1, s2: Strings to compare
        threshold: Maximum edit distance to consider a match
        
    Returns:
        True if strings are similar enough
    """
    if abs(len(s1) - len(s2)) > threshold:
        return False
    
    # Simple Levenshtein distance (optimized for short strings)
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    
    distances = range(len(s2) + 1)
    for i1, c1 in enumerate(s1):
        new_distances = [i1 + 1]
        for i2, c2 in enumerate(s2):
            if c1 == c2:
                new_distances.append(distances[i2])
            else:
                new_distances.append(1 + min(distances[i2], distances[i2 + 1], new_distances[-1]))
        distances = new_distances
    
    return distances[-1] <= threshold


def filter_and_rank_concepts(
    llm_concepts: List[str],
    metadata: DatabaseMetadata,
    top_k: int = 5
) -> List[str]:
    """
    Complete pipeline: build vocabulary, validate concepts, return top-k.
    
    Args:
        llm_concepts: Concepts extracted by LLM
        metadata: Database metadata for vocabulary
        top_k: Number of top concepts to return
        
    Returns:
        List of validated concepts, ranked by confidence
    """
    # Build schema vocabulary
    vocabulary = build_schema_vocabulary(metadata)
    
    # Validate and score concepts
    validated = validate_concepts(llm_concepts, vocabulary)
    
    # Return top-k concept strings (without scores)
    return [concept for concept, _ in validated[:top_k]]


def extract_domain_terms(metadata: DatabaseMetadata, min_frequency: int = 2) -> Set[str]:
    """
    Extract frequently occurring domain-specific terms from schema.
    These can be used as additional concepts even if not in the question.
    
    Args:
        metadata: Database metadata
        min_frequency: Minimum occurrences to consider a term
        
    Returns:
        Set of domain terms
    """
    term_counts = {}
    
    for schema in metadata.schemas.values():
        for table in schema.tables.values():
            # Count tokens from table descriptions
            if table.description:
                tokens = _tokenize(table.description)
                for token in tokens:
                    if len(token) > 4:  # Longer terms are more domain-specific
                        term_counts[token] = term_counts.get(token, 0) + 1
    
    # Filter by frequency
    domain_terms = {term for term, count in term_counts.items() 
                   if count >= min_frequency and term not in CONCEPT_BLACKLIST}
    
    return domain_terms
