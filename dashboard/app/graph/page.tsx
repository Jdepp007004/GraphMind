"use client";
import { useEffect, useState, useCallback } from "react";
import { motion } from "framer-motion";
import {
  ReactFlow,
  Node, Edge, Background, Controls, MiniMap,
  useNodesState, useEdgesState, BackgroundVariant,
  Handle, Position, NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Search, SlidersHorizontal, X } from "lucide-react";

interface AppNode { id: string; label: string; full_pkg: string; frequency: number; out_degree: number; }
interface AppEdge { source: string; target: string; count: number; probability: number; }
interface GraphData { user_id: string; n_total_apps: number; nodes: AppNode[]; edges: AppEdge[]; }

function AppNodeComp({ data }: NodeProps) {
  const freq = (data.frequency as number) || 0;
  const maxF = (data.maxFreq as number) || 1;
  const ratio = Math.min(freq / maxF, 1);
  const size = Math.max(36, Math.min(64, 36 + ratio * 28));
  return (
    <div className="flex flex-col items-center" style={{ width: size + 24 }}>
      <Handle type="target" position={Position.Top} style={{ opacity: 0, width: 0, height: 0 }} />
      <div style={{
        width: size, height: size, borderRadius: 8,
        background: `rgb(${17 + ratio * 20}, ${24 + ratio * 30}, ${39 + ratio * 50})`,
        border: `1.5px solid rgba(255,255,255,${0.1 + ratio * 0.15})`,
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: Math.max(9, size / 5), fontWeight: 600, color: "white",
        cursor: "pointer",
      }}>
        {(data.label as string).slice(0, 4).toUpperCase()}
      </div>
      <div style={{
        marginTop: 4, fontSize: 9, color: "#6b7280", maxWidth: 72,
        textOverflow: "ellipsis", overflow: "hidden", whiteSpace: "nowrap", textAlign: "center",
      }}>
        {data.label as string}
      </div>
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0, width: 0, height: 0 }} />
    </div>
  );
}

const nodeTypes = { appNode: AppNodeComp };

function buildGraph(nodes: AppNode[], edges: AppEdge[], minProb: number) {
  const top = nodes.slice(0, 20);
  const topIds = new Set(top.map(n => n.id));
  const maxF = Math.max(...top.map(n => n.frequency), 1);
  const rfNodes: Node[] = top.map((n, i) => {
    const angle = (i / top.length) * 2 * Math.PI - Math.PI / 2;
    const r = 230;
    return { id: n.id, type: "appNode", position: { x: 320 + r * Math.cos(angle), y: 260 + r * Math.sin(angle) }, data: { ...n, maxFreq: maxF } };
  });
  const rfEdges: Edge[] = edges
    .filter(e => topIds.has(e.source) && topIds.has(e.target) && e.probability >= minProb / 100)
    .slice(0, 60)
    .map(e => ({
      id: `${e.source}-${e.target}`, source: e.source, target: e.target,
      label: `${(e.probability * 100).toFixed(0)}%`,
      animated: e.probability > 0.35,
      style: { stroke: `rgba(0,0,0,${0.15 + e.probability * 0.5})`, strokeWidth: Math.max(1, e.probability * 3) },
      labelStyle: { fontSize: 9, fill: "#9ca3af" },
      labelBgStyle: { fill: "rgba(255,255,255,0.85)" },
    }));
  return { rfNodes, rfEdges };
}

export default function GraphExplorer() {
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [search, setSearch] = useState("");
  const [minProb, setMinProb] = useState(8);
  const [selected, setSelected] = useState<AppNode | null>(null);

  useEffect(() => {
    fetch("/data/graph.json").then(r => r.json()).then((d: GraphData) => {
      setGraph(d);
      const { rfNodes, rfEdges } = buildGraph(d.nodes, d.edges, minProb);
      setNodes(rfNodes); setEdges(rfEdges);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!graph) return;
    const { rfNodes, rfEdges } = buildGraph(graph.nodes, graph.edges, minProb);
    setNodes(rfNodes); setEdges(rfEdges);
  }, [minProb, graph]);

  useEffect(() => {
    if (!search) { setNodes(ns => ns.map(n => ({ ...n, style: {} }))); return; }
    setNodes(ns => ns.map(n => ({
      ...n,
      style: (n.data.label as string).toLowerCase().includes(search.toLowerCase())
        ? { outline: "2px solid #3b82f6", borderRadius: 8 }
        : { opacity: 0.25 },
    })));
  }, [search]);

  const onNodeClick = useCallback((_: any, node: Node) => {
    setSelected(graph?.nodes.find(n => n.id === node.id) || null);
  }, [graph]);

  return (
    <div className="max-w-6xl mx-auto px-8 py-10">
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mb-5">
        <h1 className="page-title mb-1">Graph Explorer</h1>
        <p className="text-sm text-gray-500">
          User <span className="mono">{graph?.user_id}</span> · {graph?.n_total_apps?.toLocaleString()} app events · Markov transition graph
        </p>
      </motion.div>

      {/* Controls */}
      <div className="card p-3.5 mb-4 flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2 flex-1 min-w-[160px]">
          <Search size={13} className="text-gray-400" />
          <input value={search} onChange={e => setSearch(e.target.value)}
            placeholder="Search app name…"
            className="input bg-transparent border-0 text-sm flex-1 p-0 focus:ring-0" style={{ border: "none", background: "transparent" }} />
          {search && <button onClick={() => setSearch("")}><X size={13} className="text-gray-400" /></button>}
        </div>
        <div className="w-px h-4 bg-gray-200" />
        <div className="flex items-center gap-2.5">
          <SlidersHorizontal size={13} className="text-gray-400" />
          <span className="text-xs text-gray-500">Min prob</span>
          <input type="range" min={1} max={30} value={minProb}
            onChange={e => setMinProb(Number(e.target.value))} className="w-20 accent-gray-700" />
          <span className="mono text-xs text-gray-700 w-8">{minProb}%</span>
        </div>
        <div className="w-px h-4 bg-gray-200" />
        <span className="text-xs text-gray-500">{edges.length} edges · {nodes.length} nodes</span>
      </div>

      <div className="grid grid-cols-4 gap-4">
        {/* Graph */}
        <div className="col-span-3 card overflow-hidden" style={{ height: 560 }}>
          <ReactFlow
            nodes={nodes} edges={edges}
            onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
            onNodeClick={onNodeClick} nodeTypes={nodeTypes}
            fitView minZoom={0.3} maxZoom={2}>
            <Background variant={BackgroundVariant.Dots} color="#e5e7eb" gap={20} size={1} />
            <Controls style={{ background: "white", border: "1px solid #e5e7eb", borderRadius: 6, boxShadow: "none" }} />
            <MiniMap nodeColor={() => "#e5e7eb"} maskColor="rgba(249,250,251,0.8)"
              style={{ background: "white", border: "1px solid #e5e7eb", borderRadius: 6 }} />
          </ReactFlow>
        </div>

        {/* Sidebar */}
        <div className="space-y-3">
          {selected ? (
            <div className="card p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="section-title">App Detail</span>
                <button onClick={() => setSelected(null)}><X size={13} className="text-gray-400" /></button>
              </div>
              <div className="space-y-2 text-xs">
                <div><div className="label mb-1">Package</div><div className="mono text-gray-700 break-all leading-relaxed">{selected.full_pkg}</div></div>
                <div><div className="label mb-1">Frequency</div><div className="font-semibold text-gray-900">{selected.frequency}</div></div>
                <div><div className="label mb-1">Out-degree</div><div className="font-semibold text-gray-900">{selected.out_degree}</div></div>
              </div>
              <div className="mt-3 pt-3" style={{ borderTop: "1px solid #f3f4f6" }}>
                <div className="label mb-2">Top outgoing edges</div>
                <div className="space-y-1.5">
                  {graph?.edges.filter(e => e.source === selected.id)
                    .sort((a, b) => b.probability - a.probability).slice(0, 6)
                    .map(e => (
                      <div key={e.target} className="flex justify-between text-xs">
                        <span className="text-gray-600 truncate flex-1">{e.target.split(".").pop()}</span>
                        <span className="mono text-gray-900 font-medium ml-2">{(e.probability * 100).toFixed(0)}%</span>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="card p-4">
              <div className="section-title mb-3">Graph Info</div>
              <div className="space-y-2 text-xs">
                {[
                  { k: "User", v: graph?.user_id || "—" },
                  { k: "Total apps", v: graph?.n_total_apps?.toLocaleString() || "—" },
                  { k: "Nodes shown", v: `${nodes.length}` },
                  { k: "Edges shown", v: `${edges.length}` },
                  { k: "Min edge prob", v: `${minProb}%` },
                ].map(item => (
                  <div key={item.k} className="flex justify-between py-1" style={{ borderBottom: "1px solid #f9fafb" }}>
                    <span className="text-gray-500">{item.k}</span>
                    <span className="mono font-medium text-gray-900">{item.v}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="card p-4">
            <div className="label mb-3">Legend</div>
            <div className="space-y-2 text-xs text-gray-600">
              <div className="flex items-center gap-2"><div style={{ width: 16, height: 16, borderRadius: 3, background: "#111827" }} /><span>High frequency app</span></div>
              <div className="flex items-center gap-2"><div style={{ width: 16, height: 16, borderRadius: 3, background: "#e5e7eb" }} /><span>Low frequency app</span></div>
              <div className="flex items-center gap-2"><div style={{ width: 24, height: 2, background: "#9ca3af" }} /><span>Low prob edge</span></div>
              <div className="flex items-center gap-2"><div style={{ width: 24, height: 3, background: "#111827" }} /><span>High prob edge</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
