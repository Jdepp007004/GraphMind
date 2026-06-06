"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  RadarChart, PolarGrid, PolarAngleAxis, Radar, Cell,
} from "recharts";
import { TrendingUp, Target, Zap, ChevronDown, ChevronUp, Info } from "lucide-react";

interface PolicyResult {
  policy: string; f1: number; std_f1: number; precision: number; recall: number;
  hit_rate: number; latency_saved_ms: number; delta_f1: number;
  p_value: number; cohen_d: number; significant: boolean; n_users: number; config: string;
}

const POLICY_COLORS: Record<string, string> = {
  "GraphMindRL_V5":       "#00e676",
  "GraphMindRL_V5_t10":   "#69f0ae",
  "RL_LatencyFocus":      "#00b0ff",
  "GraphMindRL_Baseline": "#7a8fbf",
  "Graph+Confidence":     "#90a4ae",
  "Markov2":              "#546e7a",
  "Markov1":              "#455a64",
  "GraphOnly":            "#37474f",
  "GlobalMarkov2":        "#263238",
};

const SHORT_NAMES: Record<string, string> = {
  "GraphMindRL_V5": "V5 (Prod)",
  "GraphMindRL_V5_t10": "V5 t=0.10",
  "RL_LatencyFocus": "RL Focus",
  "GraphMindRL_Baseline": "Baseline",
  "Graph+Confidence": "G+Conf",
  "Markov2": "Markov-2",
  "Markov1": "Markov-1",
  "GraphOnly": "GraphOnly",
  "GlobalMarkov2": "GlobalM2",
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass-card px-4 py-3 text-sm min-w-[180px]">
      <p className="font-semibold text-white mb-2">{label}</p>
      {payload.map((p: any) => (
        <div key={p.name} className="flex justify-between gap-4">
          <span style={{ color: p.fill || p.color }}>{p.name}</span>
          <span className="font-mono text-white">{p.value?.toFixed(4)}</span>
        </div>
      ))}
    </div>
  );
};

export default function BenchmarkExplorer() {
  const [data, setData] = useState<PolicyResult[]>([]);
  const [sortKey, setSortKey] = useState<keyof PolicyResult>("f1");
  const [sortAsc, setSortAsc] = useState(false);
  const [selected, setSelected] = useState<PolicyResult | null>(null);
  const [activeTab, setActiveTab] = useState<"f1"|"hitrate"|"latency">("f1");

  useEffect(() => {
    fetch("/data/benchmark.json").then(r => r.json()).then(setData).catch(() => {});
  }, []);

  const sorted = [...data].sort((a, b) => {
    const av = a[sortKey] as number;
    const bv = b[sortKey] as number;
    return sortAsc ? av - bv : bv - av;
  });

  const chartData = data.map(d => ({
    ...d,
    name: SHORT_NAMES[d.policy] || d.policy,
    f1_pct: d.f1 * 100,
    hit_rate_pct: d.hit_rate * 100,
  }));

  const SortHeader = ({ col, label }: { col: keyof PolicyResult; label: string }) => (
    <th className="text-left py-3 px-4 text-xs text-[#7a8fbf] font-medium cursor-pointer hover:text-white"
        onClick={() => { setSortKey(col); setSortAsc(sortKey === col ? !sortAsc : false); }}>
      <div className="flex items-center gap-1">
        {label}
        {sortKey === col ? (sortAsc ? <ChevronUp size={12} /> : <ChevronDown size={12} />) : null}
      </div>
    </th>
  );

  const tabs = [
    { key: "f1" as const, label: "F1 Score", icon: TrendingUp },
    { key: "hitrate" as const, label: "Hit Rate", icon: Target },
    { key: "latency" as const, label: "Latency Saved", icon: Zap },
  ];

  return (
    <div className="min-h-screen p-8 grid-bg">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <h1 className="text-4xl font-black mb-2">Benchmark <span className="gradient-text">Explorer</span></h1>
        <p className="text-[#7a8fbf]">31 users · 80/10/10 chronological split · UbiqLog dataset · Samsung Galaxy A23 latency</p>
      </motion.div>

      {/* Chart tabs */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
        className="glass-card p-6 mb-6">
        <div className="flex gap-2 mb-6">
          {tabs.map(t => (
            <button key={t.key} onClick={() => setActiveTab(t.key)}
              className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all
                ${activeTab === t.key ? "badge-accepted" : "text-[#7a8fbf] hover:text-white hover:bg-white/5"}`}>
              <t.icon size={14} />
              {t.label}
            </button>
          ))}
        </div>

        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={chartData} margin={{ left: 0, right: 20, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="name" tick={{ fill: "#7a8fbf", fontSize: 11 }}
                   angle={-20} textAnchor="end" height={50} />
            <YAxis tick={{ fill: "#7a8fbf", fontSize: 11 }}
                   domain={activeTab === "latency" ? [0, 2200] : [activeTab === "f1" ? 65 : 85, activeTab === "f1" ? 80 : 98]} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey={activeTab === "f1" ? "f1_pct" : activeTab === "hitrate" ? "hit_rate_pct" : "latency_saved_ms"}
                 name={activeTab === "f1" ? "F1%" : activeTab === "hitrate" ? "Hit Rate%" : "Latency ms"}
                 radius={[6, 6, 0, 0]}>
              {chartData.map((entry, i) => (
                <Cell key={i} fill={POLICY_COLORS[entry.policy] || "#546e7a"}
                      fillOpacity={entry.policy === "GraphMindRL_V5" ? 1 : 0.6} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Results Table */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
        className="glass-card overflow-hidden mb-6">
        <div className="px-6 py-4 border-b border-white/5">
          <h2 className="font-bold text-lg">Full Comparison Table</h2>
          <p className="text-xs text-[#7a8fbf]">Click column headers to sort · Click row for details</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead style={{ background: "rgba(255,255,255,0.02)" }}>
              <tr>
                <th className="text-left py-3 px-4 text-xs text-[#7a8fbf] font-medium">Policy</th>
                <SortHeader col="f1" label="F1" />
                <SortHeader col="hit_rate" label="Hit Rate" />
                <SortHeader col="latency_saved_ms" label="Lat. Saved" />
                <SortHeader col="delta_f1" label="ΔF1" />
                <SortHeader col="p_value" label="p-value" />
                <SortHeader col="cohen_d" label="Cohen d" />
                <th className="text-left py-3 px-4 text-xs text-[#7a8fbf] font-medium">Sig</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r, i) => (
                <motion.tr key={r.policy}
                  initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.04 }}
                  onClick={() => setSelected(r === selected ? null : r)}
                  className="border-t border-white/5 cursor-pointer transition-all"
                  style={{
                    background: r === selected ? "rgba(0,176,255,0.08)"
                      : r.policy === "GraphMindRL_V5" ? "rgba(0,230,118,0.05)" : "transparent"
                  }}
                  onMouseEnter={e => (e.currentTarget.style.background = "rgba(255,255,255,0.03)")}
                  onMouseLeave={e => (e.currentTarget.style.background =
                    r === selected ? "rgba(0,176,255,0.08)"
                    : r.policy === "GraphMindRL_V5" ? "rgba(0,230,118,0.05)" : "transparent")}>
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full" style={{ background: POLICY_COLORS[r.policy] || "#546e7a" }} />
                      <span className={`text-sm font-${r.policy === "GraphMindRL_V5" ? "bold text-[#00e676]" : "medium text-white"}`}>
                        {r.policy}
                      </span>
                      {r.policy === "GraphMindRL_V5" && (
                        <span className="badge-production px-2 py-0.5 rounded text-xs">PROD</span>
                      )}
                    </div>
                  </td>
                  <td className="py-3 px-4 font-mono text-sm" style={{ color: POLICY_COLORS[r.policy] }}>
                    {r.f1?.toFixed(4)}
                  </td>
                  <td className="py-3 px-4 font-mono text-sm text-[#7a8fbf]">
                    {(r.hit_rate * 100)?.toFixed(1)}%
                  </td>
                  <td className="py-3 px-4 font-mono text-sm text-[#7a8fbf]">
                    {r.latency_saved_ms?.toFixed(0)}ms
                  </td>
                  <td className="py-3 px-4 font-mono text-sm">
                    <span style={{ color: r.delta_f1 > 0 ? "#00e676" : r.delta_f1 < 0 ? "#ff5252" : "#7a8fbf" }}>
                      {r.delta_f1 > 0 ? "+" : ""}{r.delta_f1?.toFixed(4)}
                    </span>
                  </td>
                  <td className="py-3 px-4 font-mono text-sm text-[#7a8fbf]">
                    {r.p_value < 0.001 ? "<0.001" : r.p_value?.toFixed(4)}
                  </td>
                  <td className="py-3 px-4 font-mono text-sm text-[#7a8fbf]">
                    {r.cohen_d?.toFixed(3)}
                  </td>
                  <td className="py-3 px-4">
                    {r.significant
                      ? <span className="badge-production px-2 py-0.5 rounded text-xs">✓ SIG</span>
                      : <span className="badge-baseline px-2 py-0.5 rounded text-xs">n.s.</span>}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>

        {selected && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }}
            className="border-t border-white/10 p-6"
            style={{ background: "rgba(0,176,255,0.06)" }}>
            <div className="flex items-center gap-2 mb-3">
              <Info size={14} className="text-[#00b0ff]" />
              <h3 className="font-semibold text-[#00b0ff]">{selected.policy} — Configuration</h3>
            </div>
            <div className="grid grid-cols-4 gap-4">
              {[
                { k: "F1", v: selected.f1.toFixed(4) },
                { k: "Precision", v: selected.precision.toFixed(4) },
                { k: "Recall", v: selected.recall.toFixed(4) },
                { k: "Latency Saved", v: `${selected.latency_saved_ms.toFixed(0)}ms` },
              ].map(item => (
                <div key={item.k} className="text-center">
                  <div className="text-xs text-[#7a8fbf] mb-1">{item.k}</div>
                  <div className="font-mono text-lg font-bold text-white">{item.v}</div>
                </div>
              ))}
            </div>
            {selected.config && (
              <div className="mt-3 p-3 rounded-lg font-mono text-xs text-[#7a8fbf]"
                   style={{ background: "rgba(255,255,255,0.03)" }}>{selected.config}</div>
            )}
          </motion.div>
        )}
      </motion.div>

      {/* Stats footer */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }}
        className="grid grid-cols-3 gap-4">
        {[
          { label: "Baseline vs V5", value: "ΔF1 = +0.0321", sub: "t = 2.681, p = 0.0115", color: "#00e676" },
          { label: "Effect Size", value: "Cohen d = 0.491", sub: "Medium-to-large effect", color: "#00b0ff" },
          { label: "Reproducibility", value: "2 × confirmed", sub: "Identical results both runs", color: "#e040fb" },
        ].map(s => (
          <div key={s.label} className="glass-card p-4 text-center">
            <div className="text-xs text-[#7a8fbf] mb-1">{s.label}</div>
            <div className="font-bold metric-number" style={{ color: s.color }}>{s.value}</div>
            <div className="text-xs text-[#7a8fbf] mt-1">{s.sub}</div>
          </div>
        ))}
      </motion.div>
    </div>
  );
}
