"""Extract constraints from natural language questions for WHERE clause generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Any

from ..ollama_utils import generate_with_ollama


class ConstraintType(Enum):
    """Types of constraints that can be extracted from questions."""
    TEMPORAL = "temporal"  # Date/time constraints
    NOMINAL = "nominal"    # Categorical/string values
    NUMERIC = "numeric"    # Numeric comparisons
    GEOGRAPHIC = "geographic"  # Location-based


@dataclass
class Constraint:
    """Represents a constraint extracted from a question."""
    type: ConstraintType
    value: Any
    operator: str = "="  # =, >, <, >=, <=, IN, LIKE, BETWEEN
    column_hint: Optional[str] = None  # Suggested column name
    confidence: float = 1.0
    
    def __repr__(self) -> str:
        return f"Constraint({self.type.value}, {self.operator} {self.value}, col={self.column_hint})"


def extract_constraints(question: str, use_llm: bool = True) -> List[Constraint]:
    """
    Extract constraints from a natural language question.
    
    Args:
        question: The user's natural language question
        use_llm: Whether to use LLM for extraction (in addition to regex)
        
    Returns:
        List of extracted constraints
    """
    constraints: List[Constraint] = []
    
    # 1. Regex-based extraction (fast, reliable for common patterns)
    constraints.extend(_extract_temporal_regex(question))
    constraints.extend(_extract_numeric_regex(question))
    constraints.extend(_extract_nominal_regex(question))
    
    # 2. LLM-based extraction (for complex or ambiguous cases)
    if use_llm:
        llm_constraints = _extract_with_llm(question)
        # Merge, avoiding duplicates
        for llm_constraint in llm_constraints:
            if not _is_duplicate(llm_constraint, constraints):
                constraints.append(llm_constraint)
    
    return constraints


def _extract_temporal_regex(question: str) -> List[Constraint]:
    """Extract date/time constraints using regex patterns."""
    constraints = []
    
    # Pattern 1: "in July 2017", "in 2020", "in Q3 2019"
    month_year = re.search(r'\bin\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})', question, re.IGNORECASE)
    if month_year:
        month, year = month_year.groups()
        month_num = datetime.strptime(month, "%B").month
        constraints.append(Constraint(
            type=ConstraintType.TEMPORAL,
            value=f"{year}-{month_num:02d}",
            operator="LIKE",
            column_hint="date",
            confidence=0.95
        ))
    
    # Pattern 2: "in 2020", "during 2019"
    year_only = re.search(r'\b(in|during|for)\s+(\d{4})\b', question, re.IGNORECASE)
    if year_only and not month_year:  # Avoid duplicate if already found month+year
        year = year_only.group(2)
        constraints.append(Constraint(
            type=ConstraintType.TEMPORAL,
            value=year,
            operator="LIKE",
            column_hint="year",
            confidence=0.90
        ))
    
    # Pattern 3: Specific dates "2020-01-15", "01/15/2020"
    iso_date = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', question)
    if iso_date:
        constraints.append(Constraint(
            type=ConstraintType.TEMPORAL,
            value=iso_date.group(1),
            operator="=",
            column_hint="date",
            confidence=1.0
        ))
    
    # Pattern 4: Relative dates "last year", "last month", "yesterday"
    relative = re.search(r'\b(last|previous|next|this)\s+(year|month|week|day|quarter)', question, re.IGNORECASE)
    if relative:
        period = f"{relative.group(1)} {relative.group(2)}"
        constraints.append(Constraint(
            type=ConstraintType.TEMPORAL,
            value=period,
            operator="RELATIVE",
            column_hint="date",
            confidence=0.80
        ))
    
    # Pattern 5: Quarters "Q1 2020", "Q3"
    quarter = re.search(r'\bQ([1-4])\s*(\d{4})?\b', question, re.IGNORECASE)
    if quarter:
        q_num = quarter.group(1)
        year = quarter.group(2) if quarter.group(2) else "current"
        constraints.append(Constraint(
            type=ConstraintType.TEMPORAL,
            value=f"Q{q_num} {year}",
            operator="BETWEEN",
            column_hint="quarter",
            confidence=0.85
        ))
    
    return constraints


def _extract_numeric_regex(question: str) -> List[Constraint]:
    """Extract numeric constraints (>, <, =, BETWEEN)."""
    constraints = []
    
    # Pattern 1: "greater than 100", "more than 50"
    greater = re.search(r'(greater|more|higher|above|over|exceeds?)\s+than\s+(\d+(?:\.\d+)?)', question, re.IGNORECASE)
    if greater:
        constraints.append(Constraint(
            type=ConstraintType.NUMERIC,
            value=float(greater.group(2)),
            operator=">",
            confidence=0.90
        ))
    
    # Pattern 2: "less than 100", "below 50", "under 200"
    less = re.search(r'(less|fewer|lower|below|under)\s+than\s+(\d+(?:\.\d+)?)', question, re.IGNORECASE)
    if less:
        constraints.append(Constraint(
            type=ConstraintType.NUMERIC,
            value=float(less.group(2)),
            operator="<",
            confidence=0.90
        ))
    
    # Pattern 3: "equals 100", "is 50", "= 30"
    equals = re.search(r'(equals?|is|=)\s+(\d+(?:\.\d+)?)', question, re.IGNORECASE)
    if equals and not (greater or less):  # Avoid conflict
        constraints.append(Constraint(
            type=ConstraintType.NUMERIC,
            value=float(equals.group(2)),
            operator="=",
            confidence=0.85
        ))
    
    # Pattern 4: "between 10 and 100"
    between = re.search(r'between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)', question, re.IGNORECASE)
    if between:
        constraints.append(Constraint(
            type=ConstraintType.NUMERIC,
            value=(float(between.group(1)), float(between.group(2))),
            operator="BETWEEN",
            confidence=0.95
        ))
    
    # Pattern 5: "top 10", "top-5", "limit 20"
    top_n = re.search(r'\b(top|limit)\s*[-\s]*(\d+)', question, re.IGNORECASE)
    if top_n:
        constraints.append(Constraint(
            type=ConstraintType.NUMERIC,
            value=int(top_n.group(2)),
            operator="LIMIT",
            column_hint="rank",
            confidence=0.95
        ))
    
    return constraints


def _extract_nominal_regex(question: str) -> List[Constraint]:
    """Extract categorical/string constraints."""
    constraints = []
    
    # Pattern 1: Quoted strings 'Active', "Electronics"
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", question)
    for match in quoted:
        constraints.append(Constraint(
            type=ConstraintType.NOMINAL,
            value=match,
            operator="=",
            confidence=0.90
        ))
    
    # Pattern 2: Status/category patterns "status is Active", "type = npm"
    status = re.search(r'(status|type|category|state)\s+(is|=|equals?)\s+([A-Za-z0-9_-]+)', question, re.IGNORECASE)
    if status:
        column = status.group(1)
        value = status.group(3)
        constraints.append(Constraint(
            type=ConstraintType.NOMINAL,
            value=value,
            operator="=",
            column_hint=column.lower(),
            confidence=0.85
        ))
    
    # Pattern 3: Common IDs/codes "CVE-2021-12345", "ISSUE-123"
    cve = re.search(r'\b(CVE-\d{4}-\d{4,})\b', question)
    if cve:
        constraints.append(Constraint(
            type=ConstraintType.NOMINAL,
            value=cve.group(1),
            operator="=",
            column_hint="cve_id",
            confidence=1.0
        ))
    
    return constraints


def _extract_with_llm(question: str) -> List[Constraint]:
    """
    Use LLM to extract constraints that regex might miss.
    Uses a structured prompt to minimize hallucination.
    """
    prompt = f"""Extract SQL WHERE clause constraints from this question.
Output ONLY valid constraints in this format:
- Type: temporal|numeric|nominal|geographic
- Operator: =, >, <, >=, <=, IN, LIKE, BETWEEN
- Value: the actual value
- Column hint: suggested column name (optional)

Question: "{question}"

Output format (JSON list):
[{{"type": "temporal", "operator": "=", "value": "2021-07", "column_hint": "date"}}]

Constraints:"""

    try:
        response = generate_with_ollama(prompt, "qwen2.5:0.5b")
        if not response:
            return []
        
        # Try to parse JSON response
        import json
        response_clean = response.strip()
        if response_clean.startswith('['):
            constraints_data = json.loads(response_clean)
            constraints = []
            for item in constraints_data:
                try:
                    constraints.append(Constraint(
                        type=ConstraintType(item['type']),
                        value=item['value'],
                        operator=item.get('operator', '='),
                        column_hint=item.get('column_hint'),
                        confidence=0.70  # Lower confidence for LLM-extracted
                    ))
                except (KeyError, ValueError):
                    continue
            return constraints
    except Exception:
        pass
    
    return []


def _is_duplicate(constraint: Constraint, existing: List[Constraint]) -> bool:
    """Check if constraint is a duplicate of any existing constraint."""
    for exist in existing:
        if (exist.type == constraint.type and 
            exist.value == constraint.value and 
            exist.operator == constraint.operator):
            return True
    return False


def match_constraints_to_columns(
    constraints: List[Constraint],
    columns: List[dict]
) -> List[Constraint]:
    """
    Match extracted constraints to actual database columns.
    
    Args:
        constraints: List of extracted constraints
        columns: List of column metadata dicts with 'name', 'type', 'table'
        
    Returns:
        Constraints with updated column_hint based on actual schema
    """
    for constraint in constraints:
        if constraint.column_hint:
            continue  # Already has a hint
        
        # Match based on constraint type and column type
        if constraint.type == ConstraintType.TEMPORAL:
            # Find date/timestamp columns
            date_cols = [c for c in columns 
                        if any(d in c.get('type', '').lower() 
                              for d in ['date', 'time', 'timestamp'])]
            if date_cols:
                constraint.column_hint = date_cols[0]['name']
        
        elif constraint.type == ConstraintType.NUMERIC:
            # Find numeric columns
            num_cols = [c for c in columns 
                       if any(n in c.get('type', '').lower() 
                             for n in ['int', 'float', 'numeric', 'number', 'decimal'])]
            if num_cols:
                constraint.column_hint = num_cols[0]['name']
        
        elif constraint.type == ConstraintType.NOMINAL:
            # Find varchar/string columns
            str_cols = [c for c in columns 
                       if any(s in c.get('type', '').lower() 
                             for s in ['varchar', 'string', 'text', 'char'])]
            if str_cols:
                constraint.column_hint = str_cols[0]['name']
    
    return constraints
