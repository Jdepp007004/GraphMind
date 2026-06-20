"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
} from "recharts";
import { ChevronDown, ChevronUp } from "lucide-react";

interface PolicyResult {
  policy: string; f1: number; std_f1: number; precision: number; recall: number;
  hit_rate: number; latency_saved_ms: number; delta_f1: number;
  p_value: number; cohen_d: number; significant: boolean; config: string;
}

const SHORT: Record<string, string> = {
  "GraphMindRL_V5":       "V5 (Prod)",
  "GraphMindRL_V5_t10":   "V5 t=0.10",
  "RL_LatencyFocus":      "RL Focus",
  "GraphMindRL_Baseline": "Baseline",
  "Graph+Confidence":     "G+Conf",
  "Markov2":              "Markov-2",
  "Markov1":              "Markov-1",
  "GraphOnly":            "GraphOnly",
  "GlobalMarkov2":        "GlobalM2",
};

const TABS = [
  { key: "f1", label: "F1 Score" },
  { key: "hit_rate", label: "Hit Rate" },
  { key: "latency_saved_ms", label: "Latency Saved" },
] as const;

const Tip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0];
  return (
    <div className="card-sm px-3 py-2 text-xs shadow-sm">
      <div className="font-medium text-gray-900 mb-1">{d.payload.policy}</div>
      <div className="text-gray-600">{d.name}: <span className="font-semibold text-gray-900">{typeof d.value === "number" ? d.value.toFixed(4) : d.value}</span></div>
    </div>
  );
};

export default function Benchmark() {
  const [data, setData] = useState<PolicyResult[]>([]);
  const [tab, setTab] = useState<"f1" | "hit_rate" | "latency_saved_ms">("f1");
  const [sortKey, setSortKey] = useState<keyof PolicyResult>("f1");
  const [sortAsc, setSortAsc] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  useEffect(() => {
    fetch("/data/benchmark.json").then(r => r.json()).then(setData).catch(() => {});
  }, []);

  const sorted = [...data].sort((a, b) => {
    const av = a[sortKey] as number, bv = b[sortKey] as number;
    return sortAsc ? av - bv : bv - av;
  });

  const chartData = data.map(d => ({
    ...d,
    name: SHORT[d.policy] || d.policy,
    f1_pct: +(d.f1 * 100).toFixed(2),
    hit_rate_pct: +(d.hit_rate * 100).toFixed(2),
  }));

  const dataKey = tab === "f1" ? "f1_pct" : tab === "hit_rate" ? "hit_rate_pct" : "latency_saved_ms";
  const domain = tab === "f1" ? [65, 80] : tab === "hit_rate" ? [85, 96] : undefined;

  const SortTh = ({ col, label }: { col: keyof PolicyResult; label: string }) => (
    <th className="text-left py-2.5 px-4 label cursor-pointer select-none"
        onClick={() => { setSortKey(col); setSortAsc(sortKey === col ? !sortAsc : false); }}>
      <div className="flex items-center gap-1">
        {label}
        {sortKey === col ? (sortAsc ? <ChevronUp size={11} /> : <ChevronDown size={11} />) : null}
      </div>
    </th>
  );

  return (
    <div className="max-w-5xl mx-auto px-8 py-10">
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
        <h1 className="page-title mb-1">Benchmark Explorer</h1>
        <p className="text-sm text-gray-500">14 policies · 31 users · 80/10/10 chronological split · real UbiqLog data</p>
      </motion.div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3 mb-6">
        {[
          { label: "Best Cache Hit Rate", value: "97.92%", sub: "GraphMind_V6 (real UbiqLog)", green: true },
          { label: "Best F1", value: "0.4157", sub: "GraphMind_V6 · 24% above V5", green: true },
          { label: "Thrash Rate", value: "0.00%", sub: "V6 · vs 33.98% LRU", green: true },
        ].map(s => (
          <div key={s.label} className="card p-4">
            <div className="label mb-2">{s.label}</div>
            <div className={`text-xl font-semibold ${s.green ? "text-green-600" : "text-gray-900"}`}>{s.value}</div>
            <div className="text-xs text-gray-400 mt-0.5">{s.sub}</div>
          </div>
        ))}
      </div>

      {/* Chart */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}
        className="card p-5 mb-5">
        <div className="flex items-center justify-between mb-5">
          <h2 className="section-title">Policy Comparison</h2>
          <div className="flex gap-1">
            {TABS.map(t => (
              <button key={t.key} onClick={() => setTab(t.key)}
                className={`px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  tab === t.key ? "bg-gray-100 text-gray-900" : "text-gray-500 hover:text-gray-700"
                }`}>
                {t.label}
              </button>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={chartData} margin={{ left: -10, right: 10, bottom: 20 }} barSize={24}>
            <CartesianGrid vertical={false} stroke="#f3f4f6" />
            <XAxis dataKey="name" tick={{ fill: "#9ca3af", fontSize: 11 }}
                   angle={-20} textAnchor="end" height={45} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} domain={domain}
                   axisLine={false} tickLine={false} />
            <Tooltip content={<Tip />} cursor={{ fill: "#f9fafb" }} />
            <Bar dataKey={dataKey} name={tab === "f1" ? "F1%" : tab === "hit_rate" ? "Hit Rate%" : "ms"} radius={[3, 3, 0, 0]}>
              {chartData.map((e) => (
                <Cell key={e.policy}
                  fill={e.policy === "GraphMind_V6" ? "#111827"
                    : e.policy === "GraphMind_RL" ? "#6b7280" : "#e5e7eb"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Table */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08 }}
        className="card overflow-hidden">
        <div className="px-5 py-3.5" style={{ borderBottom: "1px solid #f3f4f6" }}>
          <h2 className="section-title">All Results</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead style={{ background: "#fafafa", borderBottom: "1px solid #f3f4f6" }}>
              <tr>
                <th className="text-left py-2.5 px-4 label">Policy</th>
                <SortTh col="f1" label="F1" />
                <SortTh col="hit_rate" label="Hit Rate" />
                <SortTh col="latency_saved_ms" label="Latency" />
                <SortTh col="delta_f1" label="ΔF1" />
                <SortTh col="p_value" label="p-value" />
                <SortTh col="cohen_d" label="d" />
                <th className="text-left py-2.5 px-4 label">Sig</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r) => {
                const isSelected = selected === r.policy;
                return (
                  <>
                    <tr key={r.policy}
                      className="table-row cursor-pointer"
                      style={{ background: r.policy === "GraphMindRL_V5" ? "#fafafa" : undefined }}
                      onClick={() => setSelected(isSelected ? null : r.policy)}>
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-2">
                          <span className={`dot ${r.policy === "GraphMindRL_V5" ? "dot-green" : r.policy === "GraphMindRL_Baseline" ? "dot-gray" : "dot-gray"}`} />
                          <span className={`text-sm ${r.policy === "GraphMindRL_V5" ? "font-semibold text-gray-900" : "text-gray-700"}`}>
                            {r.policy}
                          </span>
                          {r.policy === "GraphMindRL_V5" && <span className="badge badge-green text-xs">prod</span>}
                        </div>
                      </td>
                      <td className="py-3 px-4 mono text-sm font-semibold text-gray-900">{r.f1?.toFixed(4)}</td>
                      <td className="py-3 px-4 mono text-sm text-gray-600">{(r.hit_rate * 100)?.toFixed(1)}%</td>
                      <td className="py-3 px-4 mono text-sm text-gray-600">{r.latency_saved_ms?.toFixed(0)}ms</td>
                      <td className="py-3 px-4 mono text-sm" style={{ color: r.delta_f1 > 0 ? "#15803d" : r.delta_f1 < 0 ? "#b91c1c" : "#6b7280" }}>
                        {r.delta_f1 > 0 ? "+" : ""}{r.delta_f1?.toFixed(4)}
                      </td>
                      <td className="py-3 px-4 mono text-sm text-gray-600">
                        {r.p_value < 0.001 ? "<0.001" : r.p_value?.toFixed(4)}
                      </td>
                      <td className="py-3 px-4 mono text-sm text-gray-600">{r.cohen_d?.toFixed(3)}</td>
                      <td className="py-3 px-4">
                        {r.significant
                          ? <span className="badge badge-green">sig</span>
                          : <span className="badge badge-gray">n.s.</span>}
                      </td>
                    </tr>
                    {isSelected && (
                      <tr style={{ background: "#fafafa" }}>
                        <td colSpan={8} className="px-4 py-3">
                          <div className="flex gap-6 text-xs">
                            <div><span className="text-gray-400">Precision</span> <span className="font-medium text-gray-900 ml-1">{r.precision?.toFixed(4)}</span></div>
                            <div><span className="text-gray-400">Recall</span> <span className="font-medium text-gray-900 ml-1">{r.recall?.toFixed(4)}</span></div>
                            <div className="flex-1 text-gray-400 truncate">{r.config}</div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                );
              })}
            </tbody>
          </table>
        </div>
      </motion.div>
    </div>
  );
}
