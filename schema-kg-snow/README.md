# Schema Knowledge Graph for Spider2-Snow

A toolkit for constructing and querying schema knowledge graphs from Spider2-Snow databases. This project enhances text-to-SQL capabilities by providing semantic schema linking, graph-based retrieval, and agent-friendly schema representations.

## Overview

**Schema-KG-Snow** builds knowledge graphs from Snowflake database schemas to improve schema understanding for text-to-SQL tasks. It combines lexical and semantic retrieval methods to identify relevant tables and columns from natural language questions, significantly improving schema linking for SQL generation agents.

### Key Features

- 🔍 **Intelligent Schema Linking**: Hybrid retrieval combining lexical matching and semantic embeddings
- 📊 **Knowledge Graph Construction**: Builds rich graph representations of database schemas with tables, columns, and relationships
- 🤖 **Agent-Ready Exports**: Generates compact schema bundles optimized for SQL agents
- 📈 **Evaluation Framework**: Built-in evaluation against Spider2-Snow benchmark with recall metrics
- 🔌 **Flexible Metadata Sources**: Works with static JSON/DDL files or live Snowflake connections
- ⚡ **Embedding Cache**: Fast semantic retrieval with cached embeddings via Ollama

## Architecture

The system consists of several key components:

- **Metadata Loader**: Parses schema information from Spider2-Snow resource bundles
- **Graph Builder**: Constructs NetworkX graphs with schema entities and relationships
- **Schema Linker**: Baseline lexical matching using token overlap and n-grams
- **Embedding Linker**: Semantic retrieval using local embeddings (Ollama)
- **Steiner Tree**: Graph algorithms for finding minimal subgraphs connecting relevant tables
- **Exporter**: Generates JSON schema packs for downstream agent consumption

## Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) (optional, for semantic embeddings)
- Spider2-Snow benchmark data

### Setup

```bash
# Clone the repository (or ensure you have the Spider2 workspace)
cd /path/to/Spider2

# Set up Python path
export PYTHONPATH="schema-kg-snow/src:$PYTHONPATH"

# Install dependencies (if requirements.txt exists)
# pip install -r schema-kg-snow/requirements.txt

# Verify installation
python -m schema_kg_snow.cli --help
```

## Quick Start

### 1. List Available Databases

```bash
python -m schema_kg_snow.cli list
```

### 2. Inspect a Database Schema

```bash
python -m schema_kg_snow.cli inspect --database GA360
```

### 3. Build a Knowledge Graph

```bash
python -m schema_kg_snow.cli build-graph \
    --database GA360 \
    --output visualizations/GA360_graph.json
```

Outputs node/edge counts and optionally saves a node-link JSON file for visualization.

### 4. Schema Linking (Lexical)

Find relevant tables and columns for a natural language question:

```bash
python -m schema_kg_snow.cli link \
    --database GA360 \
    --question "Find the top-selling product in July 2017" \
    --top-k 5
```

### 5. Schema Linking (Semantic with Ollama)

Enhanced retrieval using embeddings:

```bash
# Start Ollama if not running
# ollama serve

export OLLAMA_URL="http://localhost:11434"  # optional, if non-default port

python -m schema_kg_snow.cli link \
    --database GA360 \
    --question "Find the top-selling product in July 2017" \
    --top-k 5 \
    --linker ollama \
    --embedding-model nomic-embed-text \
    --embedding-cache schema-kg-snow/cache
```

Embeddings are cached locally for fast subsequent queries. Use `--refresh-cache` after schema updates.

### 6. Export Agent-Friendly Schema Packs

Generate compact schema bundles for SQL agents:

```bash
python -m schema_kg_snow.cli export-pack \
    --database GA360 \
    --output-dir agent_packs/GA360
```

Creates:
- `graph.json`: Knowledge graph structure
- `tables.json`: Table and column metadata
- `summary.json`: Database statistics and overview

### 7. Evaluate Schema Linking Performance

Run evaluation against Spider2-Snow gold SQL:

```bash
# Baseline linker
python schema-kg-snow/tests/evaluate_linker.py --top-k 5

# Embedding linker
python schema-kg-snow/tests/evaluate_linker.py \
    --use-ollama \
    --embedding-cache schema-kg-snow/cache
```

Computes Recall@K by checking if gold tables/columns from SQL queries are retrieved.

## Data Sources

### Static Schema Bundles

The system primarily works with the Spider2-Snow resource bundles located at:
```
spider2-snow/resource/databases/
```

Each database has:
- `schema.json`: Table and column definitions
- DDL files: CREATE TABLE statements
- Documentation: Column descriptions and metadata

### Live Snowflake Connection (Optional)

For up-to-date schema information:

```python
from schema_kg_snow.snowflake_loader import load_database_from_snowflake

metadata = load_database_from_snowflake(
    database_name="YOUR_DATABASE",
    credentials_path="methods/spider-agent-snow/snowflake_credential.json"
)
```

Requires `snowflake-connector-python` and Spider2 Snowflake account access.

## Project Structure

```
schema-kg-snow/
├── src/schema_kg_snow/
│   ├── cli.py                  # Command-line interface
│   ├── metadata_loader.py      # Parse schema from JSON/DDL
│   ├── graph_builder.py        # Build NetworkX graphs
│   ├── schema_linker.py        # Lexical schema linking
│   ├── embedding_linker.py     # Semantic linking with embeddings
│   ├── ollama_utils.py         # Ollama embedding integration
│   ├── steiner_tree.py         # Graph algorithms
│   ├── exporter.py             # Export schema packs
│   ├── concept_extractor.py    # Extract semantic concepts
│   ├── snowflake_loader.py     # Live Snowflake metadata
│   ├── server.py               # Optional API server
│   └── visualize.py            # Graph visualization utilities
├── tests/
│   └── evaluate_linker.py      # Evaluation scripts
├── cache/                       # Cached embeddings by database
├── prediction_result.json       # Example linking results
└── README.md
```

## Methodology

### Schema Linking Pipeline

1. **Question Analysis**: Tokenize and normalize the input question
2. **Lexical Retrieval**: Match question tokens against table/column names and descriptions
3. **Semantic Retrieval** (optional): Compute embedding similarity
4. **Hybrid Scoring**: Combine lexical and semantic scores
5. **Graph Traversal**: Use graph structure to boost related entities
6. **Ranking**: Return top-K tables and columns

### Scoring Functions

- **Lexical**: Token overlap with n-gram matching
- **Semantic**: Cosine similarity of embeddings
- **Graph-based**: PageRank-style propagation through schema relationships

## Use Cases

### For SQL Agents

Schema packs provide compact, structured metadata that agents can use to:
- Select relevant tables before querying
- Understand column semantics and relationships
- Generate more accurate SQL with reduced context

### For Benchmarking

Evaluate schema linking as a standalone task:
- Measure recall of gold tables/columns
- Compare lexical vs. semantic methods
- Analyze performance across different database domains

### For Research

Study schema understanding in text-to-SQL:
- Graph-based retrieval methods
- Hybrid lexical-semantic approaches
- Schema complexity impact on SQL generation

## Related Projects

This project is part of the **Spider2.0** ecosystem:

- **[Spider2-Snow](../spider2-snow/)**: Snowflake-hosted version of Spider2 benchmark
- **[Spider-Agent-Snow](../methods/spider-agent-snow/)**: Autonomous SQL agent for Spider2-Snow
- **[Spider2-Lite](../spider2-lite/)**: BigQuery variant of Spider2

## Status

⚠️ **Work in Progress**: This project is under active development. APIs and data formats may change.

Current focus:
- [x] Basic schema linking (lexical)
- [x] Embedding-based semantic retrieval
- [x] Graph construction and traversal
- [x] CLI and evaluation framework
- [ ] Improved concept extraction
- [ ] Graph neural network approaches
- [ ] Integration with SQL generation agents
- [ ] Comprehensive documentation

## Contributing

This is a research project. If you find issues or have suggestions:
1. Check existing code and tests
2. Ensure changes align with Spider2-Snow benchmark
3. Test with multiple databases from the benchmark

## License

Part of the Spider2.0 project. See the main Spider2 repository for license details.

## Citation

If you use this work, please cite the Spider2 paper:

```bibtex
@article{spider2,
  title={Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows},
  author={[Authors]},
  year={2024}
}
```

## Contact

For questions about this schema linking toolkit, refer to the main Spider2 repository or open an issue.
