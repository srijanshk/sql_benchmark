# Knowledge Graph Driven Schema Linking for Spider2-Snow

Research implementation of a **Knowledge Graph-based schema linking pipeline** for the Spider2.0 benchmark on Snowflake databases. This project enhances text-to-SQL performance by intelligently identifying relevant tables and columns from natural language questions using hybrid retrieval methods.

## Overview

Given a natural language question, the system produces:
- **Ranked candidate tables** relevant to the question
- **Ranked candidate columns** with role annotations (join keys, filters, measures, group-by)
- **Join paths** connecting selected tables using schema relationships

The output is a structured schema-linking payload that SQL generation agents can consume for improved accuracy.

## Repository Structure

```
├── schema-kg-snow/          # 🔬 Main Implementation
│   ├── src/                 # Core schema linking pipeline
│   │   ├── metadata/        # Schema extraction and loading
│   │   ├── graph/           # Knowledge graph construction
│   │   ├── linker/          # Hybrid retrieval (lexical + semantic)
│   │   └── cli.py           # Command-line interface
│   ├── tests/               # Evaluation framework
│   ├── cache/               # Cached embeddings (~166 databases)
│   └── web/                 # Web interface for exploration
│
├── spider2-snow/            # 📊 Spider2-Snow Benchmark Data
│   ├── resource/            # Database schemas and metadata (166+ DBs)
│   ├── evaluation_suite/    # Evaluation scripts
│   └── *.jsonl              # Question datasets
│
├── methods/                 # 🤖 Reference Baseline Agents
│   ├── spider-agent-snow/   # Baseline SQL agent for Snowflake
│   └── gold-tables/         # Gold standard schema annotations
│
└── .github/instructions/    # 📋 Project guidelines
```

## Key Features

### 🔍 Hybrid Schema Linking
Combines lexical matching (BM25, n-grams) with semantic embeddings (Ollama) for robust retrieval across diverse question types.

### 📊 Knowledge Graph Construction
Builds NetworkX graphs from Snowflake schemas:
- **Nodes**: Tables, Columns, Concepts
- **Edges**: HAS_COLUMN, FOREIGN_KEY, SIMILAR_NAME
- Captures schema relationships for intelligent traversal

### 🎯 Intelligent Candidate Ranking
Multi-stage pipeline:
1. **Concept extraction** from questions
2. **Hybrid retrieval** from KG nodes
3. **Graph expansion** via relationships
4. **Join path selection** using Steiner trees
5. **Column role annotation** (keys, filters, measures, dimensions)

### 🤖 Agent-Ready Output
Exports compact, structured schema packs optimized for SQL generation:
```json
{
  "candidate_tables": [{"table": "...", "score": 0.95, "why": ["..."]}],
  "candidate_columns": [{"column": "...", "role": "filter", "score": 0.88}],
  "join_paths": [{"tables": ["..."], "joins": [...]}]
}
```

### 📈 Comprehensive Evaluation
Built-in evaluation against Spider2-Snow gold annotations with Recall@K metrics.

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) (for semantic embeddings)
- Spider2-Snow benchmark data (included in `spider2-snow/`)

### Installation

```bash
# Clone and navigate to repository
cd Spider2

# Set Python path
export PYTHONPATH="schema-kg-snow/src:$PYTHONPATH"

# Install dependencies (if not already done)
pip install -r schema-kg-snow/requirements.txt

# Optional: Install and start Ollama
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.ai/install.sh | sh
ollama serve &
ollama pull nomic-embed-text
```

### Basic Usage

**1. List available databases:**
```bash
python -m schema_kg_snow.cli list
```

**2. Inspect a database schema:**
```bash
python -m schema_kg_snow.cli inspect --database GA360
```

**3. Schema linking (lexical baseline):**
```bash
python -m schema_kg_snow.cli link \
    --database GA360 \
    --question "Find the top-selling product in July 2017" \
    --top-k 5
```

**4. Schema linking (with semantic embeddings):**
```bash
python -m schema_kg_snow.cli link \
    --database GA360 \
    --question "Find the top-selling product in July 2017" \
    --linker ollama \
    --embedding-model nomic-embed-text \
    --embedding-cache schema-kg-snow/cache
```

**5. Export schema pack for SQL agent:**
```bash
python -m schema_kg_snow.cli export-pack \
    --database GA360 \
    --output-dir agent_packs/GA360
```

### Evaluation

```bash
# Run evaluation on Spider2-Snow benchmark
python schema-kg-snow/tests/evaluate_linker.py --top-k 5

# With embeddings
python schema-kg-snow/tests/evaluate_linker.py \
    --use-ollama \
    --embedding-cache schema-kg-snow/cache
```

## Pipeline Architecture

The schema linking pipeline follows these steps:

### Step 1: Normalize Input
- Clean and normalize the question text
- Detect intent hints (aggregation, time constraints, comparisons, top-k)

### Step 2: Extract Concepts & Constraints
- Extract domain concepts (ignoring SQL keywords)
- Capture literal constraints (dates, numbers, locations)
- Normalize synonyms

### Step 3: Hybrid Retrieval
- Query BM25 lexical index
- Query semantic embedding index (Ollama)
- Retrieve candidate Table and Column nodes from KG

### Step 4: Fusion & Filtering
- Combine lexical + semantic rankings (RRF or weighted fusion)
- Filter low-quality candidates
- Retain robust "seed" nodes

### Step 5: Graph Expansion
- Expand from seeds via KG edges (1-2 hops)
- Follow `HAS_COLUMN`, `FOREIGN_KEY`, `SIMILAR_NAME` relationships
- Score expanded nodes by edge priority and hop distance

### Step 6: Join Path Selection
- Choose minimal connecting subgraph (Steiner tree)
- Prefer FK-based paths over heuristic joins
- Return ranked join paths

### Step 7: Column Role Annotation
- Classify columns by role:
  - **Join keys**: From FK relationships
  - **Filter columns**: Match constraints from question
  - **Measure columns**: Numeric metrics for aggregation
  - **Dimension columns**: Categorical for GROUP BY

### Step 8: Output Structured Payload
Return JSON with ranked tables, columns, join paths, and explanations.

## Project Status

### ✅ Implemented

- [x] Schema knowledge graph construction for 166+ databases
- [x] Lexical schema linking with BM25 and n-gram matching
- [x] Semantic linking via Ollama embeddings
- [x] Embedding cache system (pre-computed for all databases)
- [x] CLI for graph operations and schema linking
- [x] Evaluation framework with Recall@K metrics
- [x] Schema pack export for downstream agents
- [x] Web interface for exploration and debugging

### 🚧 In Development

- [ ] Advanced concept extraction from schema metadata
- [ ] Join path optimization using Steiner trees
- [ ] Column role classification improvements
- [ ] Multi-database question handling
- [ ] Performance optimization for real-time queries

### 🔮 Future Work

- [ ] Graph neural network approaches for learned embeddings
- [ ] Integration testing with SQL generation agents
- [ ] Cross-database schema linking
- [ ] Real-time schema evolution tracking
- [ ] Support for additional embedding models (OpenAI, Anthropic)
- [ ] Interactive schema exploration UI

## Documentation

- [**Schema-KG-Snow Detailed Documentation**](./schema-kg-snow/README.md) - Full API and implementation details
- [**KG Progress Tracker**](./KG_PROGRESS.md) - Development progress and research notes
- [**Snowflake Setup Guide**](./assets/Snowflake_Guideline.md) - Snowflake connection instructions

## Common Tasks

### Generate Schema Pack for Multiple Databases

```bash
# List all databases
python -m schema_kg_snow.cli list

# Process each database
for db in GA360 BASEBALL ECOMMERCE; do
    python -m schema_kg_snow.cli export-pack \
        --database $db \
        --output-dir agent_packs/$db
done
```

### Build Knowledge Graph Visualization

```bash
python -m schema_kg_snow.cli build-graph \
    --database GA360 \
    --output visualizations/GA360_graph.json
```

### Clear Embedding Cache

```bash
# Remove cached embeddings to force regeneration
rm -rf schema-kg-snow/cache/*
```

## Spider2-Snow Benchmark

The Spider2-Snow benchmark provides 166+ real-world text-to-SQL tasks across diverse domains:
- Enterprise analytics (GA360, Adobe Analytics)
- Sports and entertainment (Baseball, F1, Soccer)
- E-commerce and retail
- Healthcare and genomics
- Government and public data

Benchmark data is included in `spider2-snow/` and should be treated as read-only.

## Citation

If you use this work in your research, please cite the Spider2 paper:

```bibtex
@article{spider2,
  title={Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows},
  author={Cao, Fengyi and others},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project builds upon the Spider2.0 benchmark. See individual component licenses for details.

---

**Note**: This is a research implementation focused on knowledge graph-driven schema linking. The pipeline is designed to complement SQL generation agents by providing intelligent schema selection, reducing context size, and improving query accuracy.