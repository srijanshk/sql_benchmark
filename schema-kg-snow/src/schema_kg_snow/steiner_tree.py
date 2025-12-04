"""Steiner tree approximation for discovering join tables in schema linking."""

from __future__ import annotations

import networkx as nx
from typing import List, Set


def steiner_tree_approximation(graph: nx.DiGraph, terminals: List[str]) -> Set[str]:
    """
    Find Steiner tree connecting terminal nodes to discover join tables.
    
    Args:
        graph: NetworkX directed graph
        terminals: List of terminal node IDs (top-ranked tables)
        
    Returns:
        Set of additional nodes (join tables) needed to connect terminals
    """
    if len(terminals) < 2:
        return set()
    
    # Convert to undirected for Steiner tree algorithm
    undirected = graph.to_undirected()
    
    # Filter terminals to only include nodes that exist in graph
    valid_terminals = [t for t in terminals if t in undirected]
    
    if len(valid_terminals) < 2:
        return set()
    
    try:
        # Use NetworkX's Steiner tree approximation
        steiner_tree = nx.approximation.steiner_tree(undirected, valid_terminals)
        
        # Return nodes that weren't in original terminals (these are join tables)
        join_tables = set(steiner_tree.nodes()) - set(valid_terminals)
        
        # Filter to only table nodes
        join_tables = {n for n in join_tables if n.startswith("table:")}
        
        return join_tables
    except (nx.NetworkXError, nx.NetworkXNoPath):
        # If no path exists between terminals, return empty set
        return set()
