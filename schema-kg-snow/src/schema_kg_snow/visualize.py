"""Visualization helpers for schema knowledge graphs."""

from __future__ import annotations

from pathlib import Path
import json

import networkx as nx


def visualize_graph_pyvis(
    graph: nx.DiGraph,
    output_html: Path,
    notebook: bool = False,
) -> None:
    """Render an interactive HTML visualization using pyvis."""

    try:
        from pyvis.network import Network  # type: ignore
    except ImportError:
        _write_offline_html(graph, output_html)
        return

    net = Network(height="800px", width="100%", notebook=notebook, directed=True)
    net.force_atlas_2based()

    color_map = {
        "database": "#1f77b4",
        "schema": "#ff7f0e",
        "table": "#2ca02c",
        "column": "#d62728",
    }

    for node_id, attrs in graph.nodes(data=True):
        kind = attrs.get("kind", "node")
        title = f"{kind}: {attrs.get('name')}"
        meta_parts = [
            f"{k}: {v}"
            for k, v in attrs.items()
            if k not in {"kind", "name", "unique_id", "resource_path"}
        ]
        if meta_parts:
            title += "<br>" + "<br>".join(meta_parts)
        net.add_node(
            node_id,
            label=attrs.get("name", node_id),
            color=color_map.get(kind, "#7f7f7f"),
            title=title,
        )

    for source, target, attrs in graph.edges(data=True):
        rel = attrs.get("rel", "rel")
        net.add_edge(source, target, label=rel, arrows="to")

    try:
        net.show(str(output_html))
    except Exception:
        _write_offline_html(graph, output_html)


def _write_offline_html(graph: nx.DiGraph, output_html: Path) -> None:
    """Fallback visualization using a simple canvas layout (no external libs)."""

    color_map = {
        "database": "#1f77b4",
        "schema": "#ff7f0e",
        "table": "#2ca02c",
        "column": "#d62728",
    }
    kind_order = ["database", "schema", "table", "column"]
    spacing_x = 220
    spacing_y = 70
    margin_x = 120
    margin_y = 40

    nodes = []
    counts = {kind: 0 for kind in kind_order}
    for node_id, attrs in graph.nodes(data=True):
        kind = attrs.get("kind", "node")
        idx = kind_order.index(kind) if kind in kind_order else len(kind_order)
        x = margin_x + idx * spacing_x
        y = margin_y + counts.get(kind, 0) * spacing_y
        counts[kind] = counts.get(kind, 0) + 1
        title = f"{kind}: {attrs.get('name')}"
        meta_parts = [
            f"{k}: {v}"
            for k, v in attrs.items()
            if k not in {"kind", "name", "unique_id", "resource_path"}
        ]
        nodes.append(
            {
                "id": node_id,
                "label": attrs.get("name", node_id),
                "kind": kind,
                "color": color_map.get(kind, "#7f7f7f"),
                "title": title + "\\n" + "\\n".join(meta_parts),
                "x": x,
                "y": y,
            }
        )

    edges = []
    for source, target, attrs in graph.edges(data=True):
        edges.append(
            {
                "from": source,
                "to": target,
                "label": attrs.get("rel", ""),
            }
        )

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Schema KG</title>
  <style>
    body {{ font-family: Arial, sans-serif; }}
    #kg-canvas {{
      border: 1px solid #ccc;
      width: 100%;
      height: 800px;
    }}
    #tooltip {{
      position: absolute;
      padding: 8px;
      background: rgba(0,0,0,0.7);
      color: white;
      border-radius: 4px;
      pointer-events: none;
      display: none;
      font-size: 12px;
      max-width: 300px;
      white-space: pre-wrap;
    }}
  </style>
</head>
<body>
  <canvas id="kg-canvas" width="1200" height="800"></canvas>
  <div id="tooltip"></div>
  <script>
    const nodes = {json.dumps(nodes)};
    const edges = {json.dumps(edges)};
    const canvas = document.getElementById('kg-canvas');
    const ctx = canvas.getContext('2d');
    const tooltip = document.getElementById('tooltip');

    function draw() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.font = "12px Arial";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      edges.forEach(edge => {{
        const fromNode = nodes.find(n => n.id === edge.from);
        const toNode = nodes.find(n => n.id === edge.to);
        if (!fromNode || !toNode) return;
        ctx.strokeStyle = "#aaa";
        ctx.beginPath();
        ctx.moveTo(fromNode.x, fromNode.y);
        ctx.lineTo(toNode.x, toNode.y);
        ctx.stroke();
      }});

      nodes.forEach(node => {{
        ctx.fillStyle = node.color;
        ctx.beginPath();
        ctx.arc(node.x, node.y, 18, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#fff";
        ctx.fillText(node.label.slice(0, 12), node.x, node.y);
      }});
    }}

    function handleMouseMove(event) {{
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const hit = nodes.find(node => {{
        const dx = node.x - x;
        const dy = node.y - y;
        return Math.sqrt(dx * dx + dy * dy) <= 20;
      }});
      if (hit) {{
        tooltip.style.display = "block";
        tooltip.style.left = (event.pageX + 10) + "px";
        tooltip.style.top = (event.pageY + 10) + "px";
        tooltip.innerText = hit.title;
      }} else {{
        tooltip.style.display = "none";
      }}
    }}

    canvas.addEventListener('mousemove', handleMouseMove);
    draw();
  </script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
