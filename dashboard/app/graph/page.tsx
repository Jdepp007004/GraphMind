"use client";
import { useEffect, useState, useCallback } from "react";
import { motion } from "framer-motion";
import ReactFlow, {
  Node, Edge, Background, Controls, MiniMap,
  useNodesState, useEdgesState, BackgroundVariant,
  Handle, Position, NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Search, Filter, Play, Pause, RefreshCw } from "lucide-react";

interface AppNode { id: string; label: string; full_pkg: string; frequency: number; out_degree: number; }
interface AppEdge { source: string; target: string; count: number; probability: number; label: string; }
interface GraphData { user_id: string; n_total_apps: number; nodes: AppNode[]; edges: AppEdge[]; }

const NODE_SIZE = 40;
const TOP_N = 20;

function AppNodeComponent({ data }: NodeProps) {
  const freq = (data.frequency as number) || 0;
  const maxF = (data.maxFreq as number) || 1;
  const ratio = freq / maxF;
  const size = Math.max(32, Math.min(72, 32 + ratio * 40));
  const hue = 200 + ratio * 60;
  return (
    <div className="relative flex flex-col items-center" style={{ width: size + 20 }}>
      <Handle type="target" position={Position.Top} style={{ opacity: 0, width: 0, height: 0 }} />
      <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }}
        whileHover={{ scale: 1.1 }} transition={{ type: "spring", stiffness: 300 }}
        className="rounded-xl flex items-center justify-center font-bold text-white cursor-pointer"
        style={{
          width: size, height: size, fontSize: Math.max(8, size / 5),
          background: `hsl(${hue}, 80%, ${30 + ratio * 20}%)`,
          border: `2px solid hsl(${hue}, 80%, 50%)`,
          boxShadow: `0 0 ${10 + ratio * 15}px hsl(${hue}, 80%, 40%)`,
        }}>
        {(data.label as string).slice(0, 4).toUpperCase()}
      </motion.div>
      <div className="text-center mt-1 px-1 rounded text-[9px] max-w-[80px] truncate"
           style={{ color: `hsl(${hue}, 70%, 70%)`, background: "rgba(5,9,23,0.8)" }}>
        {data.label as string}
      </div>
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0, width: 0, height: 0 }} />
    </div>
  );
}

const nodeTypes = { appNode: AppNodeComponent };

function buildLayout(nodes: AppNode[], edges: AppEdge[], filter: number) {
  const topNodes = nodes.slice(0, Math.min(TOP_N, nodes.length));
  const topIds = new Set(topNodes.map(n => n.id));
  const maxFreq = Math.max(...topNodes.map(n => n.frequency), 1);

  // Circular layout
  const rfNodes: Node[] = topNodes.map((n, i) => {
    const angle = (i / topNodes.length) * 2 * Math.PI - Math.PI / 2;
    const r = 250;
    return {
      id: n.id,
      type: "appNode",
      position: { x: 350 + r * Math.cos(angle), y: 280 + r * Math.sin(angle) },
      data: { ...n, maxFreq },
    };
  });

  const rfEdges: Edge[] = edges
    .filter(e => topIds.has(e.source) && topIds.has(e.target) && e.probability >= filter / 100)
    .slice(0, 80)
    .map(e => ({
      id: `${e.source}-${e.target}`,
      source: e.source,
      target: e.target,
      label: e.label,
      animated: e.probability > 0.3,
      style: {
        stroke: `rgba(0, ${Math.round(100 + e.probability * 155)}, 255, ${0.3 + e.probability * 0.6})`,
        strokeWidth: Math.max(1, e.probability * 6),
      },
      labelStyle: { fontSize: 9, fill: "#7a8fbf" },
      labelBgStyle: { fill: "rgba(5,9,23,0.85)" },
    }));

  return { rfNodes, rfEdges };
}

export default function GraphExplorer() {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [search, setSearch] = useState("");
  const [minProb, setMinProb] = useState(5);
  const [selected, setSelected] = useState<AppNode | null>(null);
  const [playing, setPlaying] = useState(false);
  const [playStep, setPlayStep] = useState(0);

  useEffect(() => {
    fetch("/data/graph.json").then(r => r.json()).then((d: GraphData) => {
      setGraphData(d);
      const { rfNodes, rfEdges } = buildLayout(d.nodes, d.edges, minProb);
      setNodes(rfNodes);
      setEdges(rfEdges);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!graphData) return;
    const { rfNodes, rfEdges } = buildLayout(graphData.nodes, graphData.edges, minProb);
    setNodes(rfNodes);
    setEdges(rfEdges);
  }, [minProb, graphData]);

  // Highlight search
  useEffect(() => {
    if (!search) { setNodes(ns => ns.map(n => ({ ...n, style: {} }))); return; }
    setNodes(ns => ns.map(n => ({
      ...n,
      style: (n.data.label as string).toLowerCase().includes(search.toLowerCase()) ||
             (n.data.full_pkg as string).toLowerCase().includes(search.toLowerCase())
        ? { outline: "2px solid #00e676", borderRadius: "10px" }
        : { opacity: 0.3 },
    })));
  }, [search]);

  // Playback: highlight top transition path
  useEffect(() => {
    if (!playing || !graphData) return;
    const topEdges = [...graphData.edges].sort((a, b) => b.probability - a.probability);
    const timer = setInterval(() => {
      setPlayStep(s => {
        const edge = topEdges[s % topEdges.length];
        if (edge) {
          setEdges(es => es.map(e =>
            e.id === `${edge.source}-${edge.target}`
              ? { ...e, animated: true, style: { ...e.style, stroke: "#00e676", strokeWidth: 4 } }
              : { ...e, animated: false }
          ));
        }
        return s + 1;
      });
    }, 800);
    return () => clearInterval(timer);
  }, [playing, graphData]);

  const onNodeClick = useCallback((_: any, node: Node) => {
    const app = graphData?.nodes.find(n => n.id === node.id) || null;
    setSelected(app);
  }, [graphData]);

  return (
    <div className="min-h-screen p-8 grid-bg">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
        <h1 className="text-4xl font-black mb-2">Graph <span className="gradient-text">Explorer</span></h1>
        <p className="text-[#7a8fbf]">
          Interactive Markov transition graph · {graphData?.user_id} · {graphData?.n_total_apps?.toLocaleString()} app events
        </p>
      </motion.div>

      {/* Controls */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
        className="glass-card p-4 mb-4 flex flex-wrap gap-4 items-center">
        <div className="flex items-center gap-2 flex-1 min-w-[200px]">
          <Search size={14} className="text-[#7a8fbf]" />
          <input value={search} onChange={e => setSearch(e.target.value)}
            placeholder="Search app..." 
            className="bg-transparent text-sm text-white placeholder-[#7a8fbf] outline-none flex-1" />
        </div>
        <div className="flex items-center gap-3">
          <Filter size={14} className="text-[#7a8fbf]" />
          <span className="text-xs text-[#7a8fbf]">Min edge prob:</span>
          <input type="range" min={1} max={30} value={minProb}
            onChange={e => setMinProb(Number(e.target.value))}
            className="w-24 accent-blue-400" />
          <span className="text-sm font-mono text-[#00b0ff]">{minProb}%</span>
        </div>
        <button onClick={() => setPlaying(p => !p)}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all
            ${playing ? "badge-production" : "badge-accepted"}`}>
          {playing ? <Pause size={14} /> : <Play size={14} />}
          {playing ? "Stop Playback" : "Play Transitions"}
        </button>
        <button onClick={() => { if (graphData) { const { rfNodes, rfEdges } = buildLayout(graphData.nodes, graphData.edges, minProb); setNodes(rfNodes); setEdges(rfEdges); }}}
          className="flex items-center gap-2 px-3 py-2 rounded-xl text-sm text-[#7a8fbf] hover:text-white hover:bg-white/5">
          <RefreshCw size={14} /> Reset
        </button>
      </motion.div>

      <div className="grid grid-cols-4 gap-4">
        {/* Flow graph */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}
          className="col-span-3 glass-card overflow-hidden" style={{ height: "600px" }}>
          <ReactFlow
            nodes={nodes} edges={edges}
            onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
            onNodeClick={onNodeClick}
            nodeTypes={nodeTypes}
            fitView
            minZoom={0.3} maxZoom={2}>
            <Background variant={BackgroundVariant.Dots} color="rgba(20,40,160,0.2)" gap={24} size={1} />
            <Controls style={{ background: "rgba(10,20,48,0.9)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "10px" }} />
            <MiniMap style={{ background: "rgba(10,20,48,0.9)", border: "1px solid rgba(255,255,255,0.1)" }}
                     nodeColor={(n) => `hsl(${200 + (n.data.frequency as number || 0) * 0.001}, 80%, 40%)`} />
          </ReactFlow>
        </motion.div>

        {/* Sidebar */}
        <div className="space-y-4">
          {/* Stats */}
          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 }}
            className="glass-card p-4">
            <h3 className="font-bold text-sm mb-3 text-[#00b0ff]">Graph Stats</h3>
            {[
              { label: "Nodes shown", value: `${Math.min(TOP_N, graphData?.nodes.length || 0)}` },
              { label: "Total nodes", value: `${graphData?.nodes.length || 0}` },
              { label: "Edges shown", value: `${edges.length}` },
              { label: "Min edge prob", value: `${minProb}%` },
              { label: "User ID", value: graphData?.user_id || "—" },
            ].map(s => (
              <div key={s.label} className="flex justify-between py-1.5 border-b border-white/5 text-xs">
                <span className="text-[#7a8fbf]">{s.label}</span>
                <span className="font-mono text-white">{s.value}</span>
              </div>
            ))}
          </motion.div>

          {/* Selected node */}
          {selected && (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
              className="glass-card p-4" style={{ border: "1px solid rgba(0,176,255,0.3)" }}>
              <h3 className="font-bold text-sm mb-3 text-[#00b0ff]">Selected App</h3>
              <div className="space-y-2 text-xs">
                <div><span className="text-[#7a8fbf]">Package:</span>
                  <span className="font-mono text-white ml-2 break-all">{selected.full_pkg}</span></div>
                <div><span className="text-[#7a8fbf]">Short name:</span>
                  <span className="text-white ml-2">{selected.label}</span></div>
                <div><span className="text-[#7a8fbf]">Frequency:</span>
                  <span className="text-[#00e676] font-mono ml-2">{selected.frequency}</span></div>
                <div><span className="text-[#7a8fbf]">Out-degree:</span>
                  <span className="text-[#00b0ff] font-mono ml-2">{selected.out_degree}</span></div>
              </div>
              <div className="mt-3 text-xs text-[#7a8fbf]">
                Outgoing edges from this app:
              </div>
              <div className="space-y-1 mt-2 max-h-32 overflow-y-auto">
                {graphData?.edges.filter(e => e.source === selected.id)
                  .sort((a,b) => b.probability - a.probability)
                  .slice(0, 8)
                  .map(e => (
                    <div key={e.target} className="flex justify-between text-xs py-1 px-2 rounded"
                         style={{ background: "rgba(255,255,255,0.03)" }}>
                      <span className="text-[#7a8fbf]">{e.target.split(".").pop()}</span>
                      <span className="font-mono text-[#00b0ff]">{(e.probability * 100).toFixed(0)}%</span>
                    </div>
                  ))}
              </div>
            </motion.div>
          )}

          {/* Legend */}
          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.5 }}
            className="glass-card p-4">
            <h3 className="font-bold text-sm mb-3 text-[#7a8fbf]">Legend</h3>
            <div className="space-y-2 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ background: "hsl(200,80%,35%)" }} />
                <span className="text-[#7a8fbf]">Low frequency app</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ background: "hsl(250,80%,45%)" }} />
                <span className="text-[#7a8fbf]">High frequency app</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-0.5 w-8 bg-[#00b0ff]" />
                <span className="text-[#7a8fbf]">Low prob edge</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-1 w-8 bg-[#00e676]" />
                <span className="text-[#7a8fbf]">High prob edge</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-0.5 w-8 bg-[#00e676] animate-pulse" />
                <span className="text-[#7a8fbf]">Animated = prob &gt; 30%</span>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
