"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
} from "recharts";
import { CheckCircle2, XCircle, AlertCircle } from "lucide-react";

interface Ablation { component: string; f1: number; delta: number; note: string; }
interface WeightPt { weights: string; w_trans: number; w_rec: number; w_freq: number; f1: number; delta_f1: number; }
interface ThreshPt { threshold: number; f1: number; precision: number; recall: number; }
interface Policy { policy: string; f1: number; delta_f1: number; p_value: number; cohen_d: number; significant: boolean; }

const Tip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="card-sm px-3 py-2 text-xs shadow-sm">
      {payload.map((p: any) => (
        <div key={p.name} className="flex justify-between gap-3">
          <span className="text-gray-500">{p.name}</span>
          <span className="font-semibold text-gray-900">{typeof p.value === "number" ? p.value.toFixed(4) : p.value}</span>
        </div>
      ))}
    </div>
  );
};

export default function Research() {
  const [ablations, setAblations] = useState<Ablation[]>([]);
  const [weights, setWeights] = useState<WeightPt[]>([]);
  const [thresholds, setThresholds] = useState<ThreshPt[]>([]);
  const [benchmark, setBenchmark] = useState<Policy[]>([]);

  useEffect(() => {
    fetch("/data/ablations.json").then(r => r.json()).then(setAblations).catch(() => {});
    fetch("/data/weight_grid.json").then(r => r.json()).then(setWeights).catch(() => {});
    fetch("/data/threshold_sweep.json").then(r => r.json()).then(setThresholds).catch(() => {});
    fetch("/data/benchmark.json").then(r => r.json()).then(setBenchmark).catch(() => {});
  }, []);

  const sigRows = benchmark.filter(b => b.policy !== "GraphMindRL_Baseline");

  return (
    <div className="max-w-5xl mx-auto px-8 py-10">
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
        <h1 className="page-title mb-1">Research Validation</h1>
        <p className="text-sm text-gray-500">Reproducibility · Ablation study · Weight grid · Threshold sweep · Statistical significance</p>
      </motion.div>

      {/* Reproducibility */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.04 }}
        className="card p-5 mb-5">
        <div className="flex items-center gap-2 mb-4">
          <CheckCircle2 size={15} className="text-green-600" />
          <h2 className="section-title">Reproducibility Certificate</h2>
        </div>
        <div className="grid grid-cols-4 gap-3">
          {[
            { label: "Run 1 (Phase 11E)", value: "0.7745", sub: "2026-06-06 09:39", ok: true },
            { label: "Run 2 (verification)", value: "0.7745", sub: "2026-06-06 10:00", ok: true },
            { label: "p-value", value: "0.0115", sub: "< 0.05 ✓" },
            { label: "Cohen d", value: "0.491", sub: "medium-large" },
          ].map(s => (
            <div key={s.label} className="p-3.5 rounded-lg" style={{ background: "#f9fafb", border: "1px solid #f3f4f6" }}>
              <div className="label mb-1.5">{s.label}</div>
              <div className={`font-semibold mono text-lg ${s.ok ? "text-green-600" : "text-gray-900"}`}>{s.value}</div>
              <div className="text-xs text-gray-400 mt-0.5">{s.sub}</div>
            </div>
          ))}
        </div>
      </motion.div>

      <div className="grid grid-cols-2 gap-4 mb-4">
        {/* Ablation */}
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.07 }}
          className="card p-5">
          <h2 className="section-title mb-1">Ablation Study</h2>
          <p className="text-xs text-gray-400 mb-4">Component contribution to F1</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={ablations} layout="vertical" barSize={12} margin={{ left: 20, right: 30 }}>
              <CartesianGrid horizontal={false} stroke="#f3f4f6" />
              <XAxis type="number" domain={[0.65, 0.79]} tick={{ fill: "#9ca3af", fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="component" tick={{ fill: "#6b7280", fontSize: 9 }} width={145} axisLine={false} tickLine={false} />
              <Tooltip content={<Tip />} cursor={{ fill: "#f9fafb" }} />
              <Bar dataKey="f1" name="F1" radius={[0, 2, 2, 0]}>
                {ablations.map((a, i) => (
                  <Cell key={i} fill={a.note === "Production model" ? "#111827" : a.delta < -0.02 ? "#d1d5db" : "#e5e7eb"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Weight grid */}
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.09 }}
          className="card p-5">
          <h2 className="section-title mb-1">Phase 11A — Weight Grid</h2>
          <p className="text-xs text-gray-400 mb-4">Best: trans=0.5 rec=0.1 freq=0.4 → F1=0.7733</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={weights.slice(0, 10)} barSize={16} margin={{ left: -15, right: 10, bottom: 30 }}>
              <CartesianGrid vertical={false} stroke="#f3f4f6" />
              <XAxis dataKey="weights" tick={{ fill: "#9ca3af", fontSize: 9 }} angle={-30} textAnchor="end" height={45} axisLine={false} tickLine={false} />
              <YAxis domain={[0.73, 0.78]} tick={{ fill: "#9ca3af", fontSize: 10 }} axisLine={false} tickLine={false} />
              <Tooltip content={<Tip />} cursor={{ fill: "#f9fafb" }} />
              <Bar dataKey="f1" name="F1" radius={[2, 2, 0, 0]}>
                {weights.slice(0, 10).map((_, i) => (
                  <Cell key={i} fill={i === 0 ? "#111827" : "#e5e7eb"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Threshold sweep */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.11 }}
        className="card p-5 mb-4">
        <h2 className="section-title mb-1">Phase 11B — Threshold Sweep</h2>
        <p className="text-xs text-gray-400 mb-4">Optimal threshold = 0.16 · F1 vs threshold with V5 weights</p>
        <ResponsiveContainer width="100%" height={160}>
          <BarChart data={thresholds} barSize={22} margin={{ left: -15, right: 20 }}>
            <CartesianGrid vertical={false} stroke="#f3f4f6" />
            <XAxis dataKey="threshold" tick={{ fill: "#9ca3af", fontSize: 10 }} axisLine={false} tickLine={false} />
            <YAxis domain={[0.72, 0.77]} tick={{ fill: "#9ca3af", fontSize: 10 }} axisLine={false} tickLine={false} />
            <Tooltip content={<Tip />} cursor={{ fill: "#f9fafb" }} />
            <ReferenceLine y={0.7424} stroke="#d1d5db" strokeDasharray="4 3"
                           label={{ value: "baseline", fill: "#9ca3af", fontSize: 10 }} />
            <Bar dataKey="f1" name="F1" radius={[2, 2, 0, 0]}>
              {thresholds.map((t, i) => (
                <Cell key={i} fill={t.threshold === 0.16 ? "#111827" : "#e5e7eb"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Significance table */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.13 }}
        className="card overflow-hidden mb-4">
        <div className="px-5 py-3.5" style={{ borderBottom: "1px solid #f3f4f6" }}>
          <h2 className="section-title">Statistical Significance</h2>
          <p className="text-xs text-gray-400 mt-0.5">Paired t-test vs GraphMindRL baseline · n = 31 · α = 0.05</p>
        </div>
        <table className="w-full">
          <thead style={{ background: "#fafafa", borderBottom: "1px solid #f3f4f6" }}>
            <tr>
              {["Policy", "ΔF1", "p-value", "Cohen d", "Verdict"].map(h => (
                <th key={h} className="text-left py-2.5 px-4 label">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sigRows.map((r, i) => (
              <tr key={r.policy} className="table-row">
                <td className="py-2.5 px-4 text-sm text-gray-800">{r.policy}</td>
                <td className="py-2.5 px-4 mono text-sm" style={{ color: r.delta_f1 > 0.02 ? "#15803d" : r.delta_f1 > 0 ? "#1d4ed8" : "#b91c1c" }}>
                  {r.delta_f1 > 0 ? "+" : ""}{r.delta_f1.toFixed(4)}
                </td>
                <td className="py-2.5 px-4 mono text-sm text-gray-600">
                  {r.p_value < 0.001 ? "<0.001" : r.p_value.toFixed(4)}
                </td>
                <td className="py-2.5 px-4 mono text-sm text-gray-600">{r.cohen_d.toFixed(3)}</td>
                <td className="py-2.5 px-4">
                  {r.significant
                    ? <span className="badge badge-green flex items-center gap-1 w-fit"><CheckCircle2 size={10} />Significant</span>
                    : <span className="badge badge-gray flex items-center gap-1 w-fit"><XCircle size={10} />Not sig.</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </motion.div>

      {/* Context note */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}
        className="card p-5">
        <div className="flex items-start gap-3">
          <AlertCircle size={15} className="text-blue-500 mt-0.5 flex-shrink-0" />
          <div>
            <h2 className="section-title mb-2">Note: Context Features (W_CONTEXT = 0.00)</h2>
            <p className="text-sm text-gray-600 mb-3">
              Time-of-day and weekday features were evaluated across 4 granularities (6/12/24/48-band).
              Coverage was not the issue — 94–98% of states appeared in training. The conditional distributions
              P(next | app, time_band) add noise on 2-month datasets and require ≥12 months for stable signal.
            </p>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <div className="label mb-2">Coverage vs outcome</div>
                {[["6-band (4h slots)", "98.5%", "Hurts"], ["12-band (2h slots)", "97.6%", "Hurts"], ["24-hour", "96.3%", "Hurts"], ["48-bucket (30min)", "94.3%", "Hurts"]].map(([l, c, r]) => (
                  <div key={l} className="flex items-center gap-2 py-1">
                    <span className="badge badge-red">{r}</span>
                    <span className="text-gray-700">{l}</span>
                    <span className="text-gray-400 ml-auto">{c}</span>
                  </div>
                ))}
              </div>
              <div className="p-3 rounded-lg text-gray-600" style={{ background: "#f0f7ff", border: "1px solid #bfdbfe" }}>
                <strong className="text-blue-700">Context in dashboard:</strong> Retained for monitoring and RL state representation.
                W_CONTEXT = 0.00 applies only to the confidence scoring function, not the RL observation space.
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
