"""FastAPI server for Schema KG visualization."""

from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import networkx as nx
import json

from .metadata_loader import discover_databases, load_database_metadata
from .embedding_linker import OllamaEmbeddingLinker
from .ollama_utils import check_ollama_connection

app = FastAPI(title="Schema KG Visualizer")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RESOURCE_ROOT = Path("spider2-snow/resource/databases")
CACHE_DIR = Path("schema-kg-snow/cache")

class PredictionRequest(BaseModel):
    database: str
    question: str
    concept_model: str = "qwen2.5:0.5b"
    embedding_model: str = "nomic-embed-text"
    top_k: int = 10
    gold_tables: list[str] = []

@app.get("/")
def root():
    return {"message": "Schema KG API is running. Go to /docs for API documentation."}

@app.get("/api/databases")
def list_databases():
    """List available databases."""
    dbs = discover_databases(RESOURCE_ROOT)
    return {"databases": dbs}

@app.post("/api/predict")
def predict(req: PredictionRequest):
    """Run prediction and return graph data for visualization."""
    if not check_ollama_connection():
        raise HTTPException(status_code=503, detail="Ollama not reachable")

    try:
        metadata = load_database_metadata(RESOURCE_ROOT, req.database)
        
        # Initialize linker (this builds the graph with concepts)
        linker = OllamaEmbeddingLinker(
            metadata,
            embedding_model=req.embedding_model,
            concept_model=req.concept_model,
            cache_dir=CACHE_DIR
        )
        
        # Run linking to get scores
        result = linker.link(req.question, top_k=req.top_k)
        
        # Prepare graph data for visualization
        # We want to send the whole graph, but maybe filtered if it's too huge?
        # For now, let's send the whole graph but optimize attributes
        nodes = []
        for node_id, attrs in linker.graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "label": attrs.get("name", node_id),
                "kind": attrs.get("kind", "unknown"),
                "description": attrs.get("description", ""),
                # Add score if available from linking result (we need to access internal scores if possible, 
                # but linker.link returns sorted lists. We can map them back.)
            })
            
        # Prune graph for visualization
        # Keep:
        # 1. Top 50 final candidates
        # 2. Top 20 from each history step
        # 3. Any gold tables (if provided)
        # 4. Concepts connected to these
        
        keep_nodes = set()
        
        # 1. Final candidates
        for t, _ in result.candidate_tables[:50]:
            keep_nodes.add(t)
            
        # 2. History steps
        if result.score_history:
            for step in result.score_history:
                # Get top 20 for this step
                top_step = sorted(step["scores"].items(), key=lambda x: x[1], reverse=True)[:20]
                for n, _ in top_step:
                    keep_nodes.add(n)
                    
        # 3. Gold tables (passed in request)
        if req.gold_tables:
             # gold_tables are just table names (e.g. "GAMES"), but IDs are "table:DB.SCHEMA.TABLE"
             # We need to find matching nodes
             for node in linker.graph.nodes:
                 if linker.graph.nodes[node].get("kind") == "table":
                     t_name = node.split(".")[-1]
                     if t_name in req.gold_tables:
                         keep_nodes.add(node)

        # 4. Add concepts that connect to these tables
        # Improve connectivity: Include 1-hop neighbors for all kept nodes
        # This ensures we see the relationships (Foreign Keys, Concepts) that connect them
        
        initial_nodes = list(keep_nodes)
        for node in initial_nodes:
            # Add neighbors (both incoming and outgoing)
            # Limit to avoid explosion? Maybe just concepts and tables?
            for neighbor in linker.graph.neighbors(node):
                keep_nodes.add(neighbor)
            for predecessor in linker.graph.predecessors(node):
                keep_nodes.add(predecessor)
        
        # Create subgraph
        subgraph = linker.graph.subgraph(keep_nodes)
        
        nodes = [{"id": n, "label": n, "kind": subgraph.nodes[n].get("kind", "unknown")} for n in subgraph.nodes]
        links = [{"source": u, "target": v, "rel": d.get("rel", "related")} for u, v, d in subgraph.edges(data=True)]
            
        # Map scores for highlighting
        scores = {}
        for t, s in result.candidate_tables:
            scores[t] = s
        for c, s in result.candidate_columns:
            scores[c] = s

        return {
            "graph": {"nodes": nodes, "links": links},
            "results": {
                "tables": [{"id": n, "score": s} for n, s in result.candidate_tables],
                "columns": [{"id": n, "score": s} for n, s in result.candidate_columns]
            },
            "scores": scores,
            "history": result.score_history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/gold_questions/{database}")
def list_gold_questions(database: str):
    """List available gold questions for a database."""
    try:
        # Load tasks from spider2-snow/spider2-snow.jsonl
        gold_path = Path("spider2-snow/spider2-snow.jsonl")
        if not gold_path.exists():
            # Try alternate location
            gold_path = Path("../spider2-snow/spider2-snow.jsonl")
        
        if not gold_path.exists():
            return {"questions": []}
            
        tasks = []
        with gold_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                task = json.loads(line)
                if task.get("db_id") == database:
                    instance_id = task.get("instance_id", "")
                    # Construct path to gold SQL file
                    # Try standard structure: spider2-snow/evaluation_suite/gold/sql/{instance_id}.sql
                    sql_path = Path("spider2-snow/evaluation_suite/gold/sql") / f"{instance_id}.sql"
                    if not sql_path.exists():
                         # Try relative to parent if running from subdir
                         sql_path = Path("../spider2-snow/evaluation_suite/gold/sql") / f"{instance_id}.sql"
                    
                    gold_tables = _parse_gold_tables(sql_path)
                    
                    tasks.append({
                        "id": instance_id,
                        "question": task.get("instruction", ""), # Map instruction to question
                        "gold_tables": gold_tables
                    })
        return {"questions": tasks}
    except Exception as e:
        print(f"Error loading gold questions: {e}")
        return {"questions": []}

def _parse_gold_tables(sql_path: Path) -> List[str]:
    """Helper to parse gold tables from SQL file."""
    tables = []
    if not sql_path.exists():
        return tables
        
    try:
        with sql_path.open("r", encoding="utf-8") as f:
            sql_text = f.read()

        # Look for fully qualified names: DB.SCHEMA.TABLE
        fq_pattern = re.compile(r'([A-Za-z0-9_"]+)\.([A-Za-z0-9_"]+)\.([A-Za-z0-9_"]+)')
        for match in fq_pattern.findall(sql_text):
            table = match[2].replace('"', "").upper()
            tables.append(table)
            
        # Fallback: SCHEMA.TABLE
        if not tables:
            short_pattern = re.compile(r'([A-Za-z0-9_"]+)\.([A-Za-z0-9_"]+)')
            for match in short_pattern.findall(sql_text):
                table = match[1].replace('"', "").upper()
                tables.append(table)
                
        return sorted(set(tables))
    except Exception:
        return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
