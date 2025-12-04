# Spider2 Text-to-SQL Research

Research workspace for working with the Spider2.0 benchmark, focusing on knowledge graph-based schema linking for Snowflake databases.

## Repository Structure

```
├── schema-kg-snow/          # 🔬 Knowledge Graph Schema Linking (Main Project)
│   ├── src/                 # Core implementation
│   ├── tests/               # Evaluation scripts
│   └── cache/               # Cached embeddings
│
├── spider2-snow/            # 📊 Spider2-Snow Benchmark (Submodule)
│   ├── resource/            # Database schemas and metadata
│   ├── evaluation_suite/    # Evaluation tools
│   └── *.jsonl              # Benchmark datasets
│
├── methods/                 # 🤖 Reference Methods & Agents
│   ├── spider-agent-snow/   # SQL agent for Snowflake
│   ├── spider-agent-lite/   # SQL agent for BigQuery
│   └── gold-tables/         # Gold standard annotations
│
└── assets/                  # 📚 Documentation & Guidelines
```

## Main Project: Schema-KG-Snow

**Schema Knowledge Graph for Spider2-Snow** is a toolkit for intelligent schema linking in text-to-SQL tasks. It builds knowledge graphs from database schemas and uses hybrid lexical-semantic retrieval to identify relevant tables and columns from natural language questions.

### Key Features

- 🔍 **Hybrid Schema Linking**: Combines lexical matching with semantic embeddings (Ollama)
- 📊 **Graph-Based Retrieval**: Leverages schema relationships for improved accuracy
- 🤖 **Agent Integration**: Exports compact schema packs for SQL generation agents
- 📈 **Evaluation Framework**: Built-in metrics against Spider2-Snow benchmark

[**→ Full Documentation**](./schema-kg-snow/README.md)

### Quick Start

```bash
# Set up environment
export PYTHONPATH="schema-kg-snow/src:$PYTHONPATH"

# Link schema to a question
python -m schema_kg_snow.cli link \
    --database GA360 \
    --question "Find the top-selling product in July 2017" \
    --top-k 5

# With semantic embeddings
python -m schema_kg_snow.cli link \
    --database GA360 \
    --question "Find the top-selling product in July 2017" \
    --linker ollama \
    --embedding-model nomic-embed-text \
    --embedding-cache schema-kg-snow/cache
```

## Spider2-Snow Benchmark

The Spider2-Snow benchmark provides real-world text-to-SQL tasks on Snowflake databases. This repository currently includes the benchmark data directly, but it can be converted to a git submodule for easier updates.

### Option 1: Current Setup (Direct Copy)

The `spider2-snow/` directory contains a copy of the benchmark data.

**Pros**: 
- Simple, everything in one place
- No git submodule complexity

**Cons**:
- Need to manually sync updates
- Duplicate data if working on multiple forks

### Option 2: Git Submodule (Recommended for Collaboration)

Convert to a submodule to track the official Spider2 repository:

```bash
# Remove current spider2-snow directory (backup first if needed)
mv spider2-snow spider2-snow.backup

# Add as submodule
git submodule add https://github.com/xlang-ai/Spider2.git spider2-official
ln -s spider2-official/spider2-snow spider2-snow

# Or point to specific path
git submodule add https://github.com/xlang-ai/Spider2.git

# Update submodule
git submodule update --init --recursive
git submodule update --remote
```

**Pros**:
- Always sync with official benchmark
- Smaller repository size
- Clear separation of your work vs. benchmark data

**Cons**:
- Requires understanding of git submodules
- Extra setup step for collaborators

### Recommendation

For this research project, I recommend **keeping the current direct copy** because:

1. You're actively developing schema linking methods that depend on the data structure
2. The benchmark is relatively stable
3. Simpler workflow for experimentation
4. You can periodically pull updates manually if needed

If you plan to contribute back to the main Spider2 project or collaborate with others who are also extending Spider2, then switching to a submodule would be beneficial.

## Development Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) (for semantic embeddings)
- Docker (for Spider agents)
- Snowflake account (optional, for live schema access)

### Installation

```bash
# Clone this repository
git clone <your-repo-url>
cd Spider2

# Set Python path
export PYTHONPATH="schema-kg-snow/src:$PYTHONPATH"

# Install Ollama (optional, for embeddings)
# Follow instructions at https://ollama.ai/

# Pull embedding model
ollama pull nomic-embed-text
```

### Running Tests

```bash
# Evaluate schema linking
python schema-kg-snow/tests/evaluate_linker.py --top-k 5

# With embeddings
python schema-kg-snow/tests/evaluate_linker.py \
    --use-ollama \
    --embedding-cache schema-kg-snow/cache
```

## Project Status

### ✅ Completed

- [x] Schema knowledge graph construction
- [x] Lexical schema linking with n-gram matching
- [x] Semantic linking via Ollama embeddings
- [x] Embedding cache system
- [x] CLI for graph building and linking
- [x] Evaluation framework against Spider2-Snow
- [x] Schema pack export for agents

### 🚧 In Progress

- [ ] Improved concept extraction from schema metadata
- [ ] Graph neural network approaches
- [ ] Integration with SQL generation agents
- [ ] Performance optimization for large schemas
- [ ] Better handling of multi-schema databases

### 🔮 Future Work

- [ ] Support for other embedding models
- [ ] Real-time schema evolution tracking
- [ ] Cross-database schema linking
- [ ] Interactive schema exploration UI

## Repository Organization

### Why This Structure?

- **schema-kg-snow/**: Your original contribution - the KG-based schema linking system
- **spider2-snow/**: Benchmark data needed for evaluation (consider as read-only)
- **methods/**: Reference implementations for comparison (from Spider2 project)

### Working with the Benchmark Data

The Spider2-Snow benchmark includes:
- 166+ databases from diverse domains
- Schema files (JSON and DDL)
- Natural language questions
- Gold SQL queries
- Evaluation scripts

Your schema-kg-snow project enhances this by providing intelligent schema retrieval before SQL generation.

## Common Tasks

### Generate Schema Pack for a Database

```bash
python -m schema_kg_snow.cli export-pack \
    --database YOUR_DATABASE \
    --output-dir agent_packs/YOUR_DATABASE
```

### Visualize Schema Graph

```bash
python -m schema_kg_snow.cli build-graph \
    --database YOUR_DATABASE \
    --output visualizations/YOUR_DATABASE.json
```

### Batch Process Multiple Databases

```bash
# List all databases
python -m schema_kg_snow.cli list

# Process each one
for db in $(python -m schema_kg_snow.cli list); do
    python -m schema_kg_snow.cli export-pack \
        --database $db \
        --output-dir agent_packs/$db
done
```

## Related Resources

- [Spider2 Official Repository](https://github.com/xlang-ai/Spider2)
- [Spider2-Snow Benchmark](./spider2-snow/README.md)
- [Snowflake Setup Guide](./assets/Snowflake_Guideline.md)
- [Schema-KG Documentation](./schema-kg-snow/README.md)

## Citation

If you use this work, please cite the Spider2 paper:

```bibtex
@article{spider2,
  title={Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows},
  author={[Authors]},
  year={2024}
}
```

## License

This research project builds upon Spider2.0. See individual component licenses for details.

## Notes

- The schema-kg-snow project is experimental and under active development
- Benchmark data (spider2-snow) should be treated as read-only
- Cache files can be safely deleted; they'll be regenerated on next use
- Embedding cache can grow large; clean periodically with `rm -rf schema-kg-snow/cache/*`
