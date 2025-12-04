"""Extract business concepts from table metadata using LLM."""

from __future__ import annotations

import re
from typing import List

from .metadata_loader import TableMetadata
from .ollama_utils import generate_with_ollama


def extract_concepts_from_table(
    table: TableMetadata, model: str = "llama3"
) -> List[str]:
    """Extract high-level business concepts from a table's metadata."""
    
    prompt = f"""
    Analyze the following database table and extract 3-5 high-level business concepts or entities it represents.
    Focus on the business domain (e.g., "Customer", "Sales", "Inventory", "Web Analytics").
    Do not just repeat the table name.
    Output ONLY a comma-separated list of concepts.

    Table: {table.name}
    Schema: {table.schema}
    Description: {table.description or "N/A"}
    Columns: {", ".join(c.name for c in table.columns[:10])}
    """

    response = generate_with_ollama(prompt, model)
    if not response:
        return []

    # Clean up response
    concepts = []
    for part in response.split(","):
        clean = re.sub(r"[^a-zA-Z0-9\s]", "", part).strip()
        if clean and len(clean) < 30:
            concepts.append(clean)
    
    return list(set(concepts))
