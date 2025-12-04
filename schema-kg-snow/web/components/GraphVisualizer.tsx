"use client";

import dynamic from "next/dynamic";
import { useRef, useState, useEffect, useMemo, useCallback } from "react";

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
});

interface Node {
  id: string;
  label: string;
  kind: string;
  description?: string;
  score?: number;
  val?: number; // For node size
  color?: string;
}

interface Link {
  source: string;
  target: string;
  rel: string;
}

interface GraphData {
  nodes: Node[];
  links: Link[];
}

interface GraphVisualizerProps {
  data: GraphData;
  scores?: Record<string, number>;
  history?: Array<{ step: string; scores: Record<string, number> }>;
  goldTables?: string[];
}

export default function GraphVisualizer({ data, scores, history, goldTables }: GraphVisualizerProps) {
  const graphRef = useRef<any>();
  const [currentStep, setCurrentStep] = useState<number>(history ? history.length - 1 : 0);

  // Update current step when history changes (new prediction)
  useEffect(() => {
    if (history && history.length > 0) {
      setCurrentStep(history.length - 1);
    }
  }, [history]);

  // Determine which scores to use
  const activeScores = useMemo(() => {
    return history && history.length > 0 
      ? history[currentStep].scores 
      : (scores || {});
  }, [history, scores, currentStep]);

  // Accessor functions for node visuals
  const getNodeVal = useCallback((node: any) => {
    const score = activeScores[node.id] || 0;
    let val = 1;
    if (node.kind === "concept") val = 3;
    else if (node.kind === "table") val = 2;
    
    if (score > 0) val += score * 10;

    // Gold table highlighting (size boost)
    if (goldTables && goldTables.length > 0 && node.kind === "table") {
      const tableName = node.id.split('.').pop();
      const isGold = goldTables.some(gt => gt === tableName || node.id.includes(gt));
      if (isGold) val = Math.max(val, 5);
      if (isGold && score === 0) val = 4;
    }
    return val;
  }, [activeScores, goldTables]);

  const getNodeColor = useCallback((node: any) => {
    const score = activeScores[node.id] || 0;
    let color = "#ccc";
    
    if (node.kind === "concept") color = "#f59e0b";
    else if (node.kind === "table") color = "#3b82f6";
    else if (node.kind === "column") color = "#10b981";

    // Gold table highlighting
    if (goldTables && goldTables.length > 0 && node.kind === "table") {
      const tableName = node.id.split('.').pop();
      const isGold = goldTables.some(gt => gt === tableName || node.id.includes(gt));
      if (isGold) {
        color = "#22c55e"; // Green
        if (score === 0) color = "#ef4444"; // Red (Missed)
      }
    }
    return color;
  }, [activeScores, goldTables]);

  // Zoom to fit only when data changes (new prediction), not on step change
  useEffect(() => {
    if (graphRef.current) {
      graphRef.current.d3Force('charge').strength(-100);
      setTimeout(() => graphRef.current.zoomToFit(400, 50), 500);
    }
  }, [data]);

  return (
    <div className="flex flex-col gap-4">
      <div className="border rounded-lg overflow-hidden h-[600px] bg-slate-50 relative">
        <ForceGraph2D
          ref={graphRef}
          graphData={data} 
          nodeLabel={(node: any) => `${node.label} (${(getNodeVal(node) - (node.kind==='concept'?3:node.kind==='table'?2:1))/10})`} 
          nodeColor={getNodeColor}
          nodeVal={getNodeVal}
          nodeRelSize={6}
          linkColor={(link: any) => {
            if (link.rel === "FOREIGN_KEY") return "#ef4444"; // Red
            if (link.rel === "RELATED_TO") return "#f59e0b"; // Amber
            return "#e2e8f0"; // Gray
          }}
          linkDirectionalArrowLength={3.5}
          linkDirectionalArrowRelPos={1}
          width={800}
          height={600}
          cooldownTicks={100}
        />
        
        {/* Step Overlay */}
        {history && history.length > 0 && (
          <div className="absolute top-4 left-4 bg-white/90 p-4 rounded shadow-lg border border-slate-200 max-w-xs">
            <h3 className="font-bold text-sm text-slate-500 mb-1">Reasoning Step {currentStep + 1}/{history.length}</h3>
            <div className="text-lg font-semibold text-slate-800 mb-2">
              {history[currentStep].step === "Steiner Tree" ? "Generated Schema" : history[currentStep].step}
            </div>
            <input 
              type="range" 
              min="0" 
              max={history.length - 1} 
              value={currentStep} 
              onChange={(e) => setCurrentStep(parseInt(e.target.value))}
              className="w-full accent-blue-600"
            />
            <div className="flex justify-between mt-2">
              <button 
                onClick={() => setCurrentStep(Math.max(0, currentStep - 1))}
                disabled={currentStep === 0}
                className="px-2 py-1 text-xs bg-slate-100 rounded hover:bg-slate-200 disabled:opacity-50"
              >
                Prev
              </button>
              <button 
                onClick={() => setCurrentStep(Math.min(history.length - 1, currentStep + 1))}
                disabled={currentStep === history.length - 1}
                className="px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200 disabled:opacity-50"
              >
                Next
              </button>
            </div>
          </div>
        )}
        {/* Legend */}
        <div className="absolute bottom-4 right-4 bg-white/90 p-3 rounded shadow-lg border border-slate-200 text-xs">
          <h4 className="font-bold text-slate-500 mb-2 uppercase tracking-wider">Legend</h4>
          <div className="space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-blue-500"></span>
              <span className="text-slate-700">Table</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-amber-500"></span>
              <span className="text-slate-700">Concept (Business Term)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-emerald-500"></span>
              <span className="text-slate-700">Column</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full bg-red-500"></span>
              <span className="text-slate-700">Foreign Key</span>
            </div>
            <div className="mt-2 pt-2 border-t border-slate-200 text-slate-500 italic">
              Node size indicates relevance score
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
