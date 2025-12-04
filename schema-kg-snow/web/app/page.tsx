"use client";

import { useState, useEffect } from "react";
import GraphVisualizer from "../components/GraphVisualizer";
import { Play, Database, Search, Loader2, List as ListIcon, Network } from "lucide-react";

interface GraphData {
  graph: {
    nodes: any[];
    links: any[];
  };
  results: {
    tables: any[];
    columns: any[];
  };
  scores: Record<string, number>;
  history?: Array<{ step: string; scores: Record<string, number> }>;
}

export default function Home() {
  const [databases, setDatabases] = useState<string[]>([]);
  const [selectedDb, setSelectedDb] = useState("");
  const [question, setQuestion] = useState("Find the top-selling product in July 2017");
  const [conceptModel, setConceptModel] = useState("qwen2.5:0.5b");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const [goldQuestions, setGoldQuestions] = useState<any[]>([]);
  const [selectedGoldId, setSelectedGoldId] = useState<string>("");
  const [goldTables, setGoldTables] = useState<string[]>([]);

  useEffect(() => {
    fetch("http://localhost:8000/api/databases")
      .then((res) => res.json())
      .then((data) => setDatabases(data.databases));
  }, []);

  // Fetch gold questions when DB changes
  useEffect(() => {
    if (selectedDb) {
      fetch(`http://localhost:8000/api/gold_questions/${selectedDb}`)
        .then((res) => res.json())
        .then((data) => setGoldQuestions(data.questions));
      setSelectedGoldId("");
      setGoldTables([]);
    }
  }, [selectedDb]);

  const handleGoldSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value;
    setSelectedGoldId(id);
    const task = goldQuestions.find(q => q.id === id);
    if (task) {
      setQuestion(task.question);
      setGoldTables(task.gold_tables);
    } else {
      setGoldTables([]);
    }
  };

  const handlePredict = async () => {
    if (!selectedDb || !question) return;

    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          database: selectedDb,
          question,
          concept_model: conceptModel,
          embedding_model: "nomic-embed-text",
          top_k: 10,
          gold_tables: goldTables
        }),
      });
      const data = await res.json();
      setResult(data);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 p-8 font-sans text-slate-900">
      <div className="max-w-6xl mx-auto">
        <header className="mb-8 flex items-center gap-3">
          <div className="p-3 bg-blue-600 rounded-lg shadow-lg shadow-blue-200">
            <Database className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold tracking-tight text-slate-900">Schema Knowledge Graph</h1>
            <p className="text-slate-500">GraphRAG Visualization & Inference</p>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Controls */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Database</label>
                <select
                  className="w-full p-2 rounded border border-slate-300 bg-slate-50 focus:ring-2 focus:ring-blue-500 outline-none transition-all"
                  value={selectedDb}
                  onChange={(e) => setSelectedDb(e.target.value)}
                >
                  <option value="">Select Database...</option>
                  {databases.map((db) => (
                    <option key={db} value={db}>
                      {db}
                    </option>
                  ))}
                </select>
              </div>

              {goldQuestions.length > 0 && (
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">
                    Load Gold Question ({goldQuestions.length})
                  </label>
                  <select
                    className="w-full p-2 rounded border border-slate-300 bg-slate-50 focus:ring-2 focus:ring-blue-500 outline-none transition-all text-sm"
                    value={selectedGoldId}
                    onChange={handleGoldSelect}
                  >
                    <option value="">-- Custom Question --</option>
                    {goldQuestions.map((q) => (
                      <option key={q.id} value={q.id}>
                        {q.id}: {q.question.substring(0, 40)}...
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <div>
                <label className="block text-sm font-medium text-slate-700 mb-1">Question</label>
                <textarea
                  className="w-full p-3 rounded border border-slate-300 bg-slate-50 focus:ring-2 focus:ring-blue-500 outline-none transition-all h-32 resize-none"
                  placeholder="Ask a question about the schema..."
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Concept Model</label>
                  <select
                    className="w-full p-2 rounded border border-slate-300 bg-slate-50 text-sm"
                    value={conceptModel}
                    onChange={(e) => setConceptModel(e.target.value)}
                  >
                    <option value="qwen2.5:0.5b">qwen2.5:0.5b (Fast)</option>
                    <option value="qwen2.5:1.5b">qwen2.5:1.5b (Balanced)</option>
                  </select>
                </div>
              </div>

              <button
                onClick={handlePredict}
                disabled={loading || !selectedDb || !question}
                className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium shadow-md shadow-blue-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-all"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" /> Processing...
                  </>
                ) : (
                  <>
                    <Play className="w-5 h-5" /> Run Prediction
                  </>
                )}
              </button>
            </div>

            {/* Results List */}
            {result && (
              <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
                <h3 className="font-semibold text-slate-900 mb-4 flex items-center gap-2">
                  <ListIcon className="w-5 h-5 text-slate-400" /> Top Candidates
                </h3>
                
                <div className="space-y-6">
                  {/* Predicted Tables */}
                  <div>
                    <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Predicted Tables</h4>
                    <div className="space-y-2">
                      {result.results.tables.slice(0, 10).map((t: any) => {
                        const tableName = t.id.split('.').pop();
                        const isGold = goldTables.some(gt => gt === tableName || t.id.includes(gt));
                        return (
                          <div key={t.id} className={`flex justify-between items-center text-sm p-2 rounded ${isGold ? 'bg-green-50 border border-green-200' : 'bg-slate-50'}`}>
                            <span className={`font-medium ${isGold ? 'text-green-700' : 'text-slate-700'}`}>
                              {t.id.split('.').pop()}
                              {isGold && <span className="ml-2 text-xs bg-green-200 text-green-800 px-1 rounded">GOLD</span>}
                            </span>
                            <span className="text-slate-400 font-mono text-xs">{t.score.toFixed(4)}</span>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Gold Standard List */}
                  {goldTables.length > 0 && (
                    <div>
                      <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-green-500"></span>
                        Gold Standard ({goldTables.length})
                      </h4>
                      <div className="space-y-2">
                        {goldTables.map((gt) => {
                          // Check if this gold table was found in top candidates
                          const found = result.results.tables.some((t: any) => t.id.includes(gt));
                          return (
                            <div key={gt} className={`flex justify-between items-center text-sm p-2 rounded border ${found ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
                              <span className={`font-medium ${found ? 'text-green-700' : 'text-red-700'}`}>
                                {gt}
                              </span>
                              <span className="text-xs font-medium px-1.5 py-0.5 rounded bg-white/50">
                                {found ? 'FOUND' : 'MISSED'}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Visualization */}
          <div className="lg:col-span-2">
            <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 h-full min-h-[600px]">
              <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Search className="w-5 h-5 text-slate-400" /> Knowledge Graph
              </h2>
              {result ? (
                <GraphVisualizer 
                  data={result.graph} 
                  scores={result.scores} 
                  history={result.history} 
                  goldTables={goldTables}
                />
              ) : (
                <div className="h-full flex flex-col items-center justify-center text-slate-400 space-y-4">
                  <div className="w-16 h-16 rounded-full bg-slate-100 flex items-center justify-center">
                    <Network className="w-8 h-8 text-slate-300" />
                  </div>
                  <p>Select a database and run prediction to visualize the schema graph</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
