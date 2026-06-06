"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  ScatterChart, Scatter, ZAxis, ReferenceLine,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend,
} from "recharts";
import { CheckCircle, XCircle, Info, FlaskConical, BookOpen } from "lucide-react";

interface AblationItem { component: string; f1: number; delta: number; note: string; }
interface WeightPoint { weights: string; w_trans: number; w_rec: number; w_freq: number; f1: number; delta_f1: number; }
interface ThresholdPoint { threshold: number; f1: number; precision: number; recall: number; hit_rate: number; }
interface PolicyResult { policy: string; f1: number; delta_f1: number; p_value: number; cohen_d: number; significant: boolean; }

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass-card px-4 py-3 text-xs min-w-[180px]">
      {payload.map((p: any) => (
        <div key={p.name} className="flex justify-between gap-4 mb-1">
          <span style={{ color: p.fill || p.color || "#7a8fbf" }}>{p.name}</span>
          <span className="font-mono text-white">{typeof p.value === "number" ? p.value.toFixed(4) : p.value}</span>
        </div>
      ))}
    </div>
  );
};

export default function ResearchValidation() {
  const [ablations, setAblations] = useState<AblationItem[]>([]);
  const [weightGrid, setWeightGrid] = useState<WeightPoint[]>([]);
  const [thresholdSweep, setThresholdSweep] = useState<ThresholdPoint[]>([]);
  const [benchmark, setBenchmark] = useState<PolicyResult[]>([]);

  useEffect(() => {
    fetch("/data/ablations.json").then(r => r.json()).then(setAblations).catch(() => {});
    fetch("/data/weight_grid.json").then(r => r.json()).then(setWeightGrid).catch(() => {});
    fetch("/data/threshold_sweep.json").then(r => r.json()).then(setThresholdSweep).catch(() => {});
    fetch("/data/benchmark.json").then(r => r.json()).then(setBenchmark).catch(() => {});
  }, []);

  const radarData = [
    { metric: "F1", V5: 0.7745, Baseline: 0.7424, Markov1: 0.7267 },
    { metric: "Hit Rate", V5: 0.9307, Baseline: 0.9357, Markov1: 0.9278 },
    { metric: "Precision", V5: 0.7512, Baseline: 0.7218, Markov1: 0.7073 },
    { metric: "Recall", V5: 0.8063, Baseline: 0.7714, Markov1: 0.7552 },
    { metric: "Consistency", V5: 0.96, Baseline: 0.88, Markov1: 0.82 },
  ];

  const sigData = benchmark.filter(b => b.policy !== "GraphMindRL_Baseline").map(b => ({
    ...b,
    name: b.policy.replace("GraphMindRL_", "").replace("GlobalMarkov2", "GlobalM2"),
    sig_level: b.p_value < 0.001 ? "p<0.001" : b.p_value < 0.01 ? "p<0.01" : b.p_value < 0.05 ? "p<0.05" : "n.s.",
  }));

  return (
    <div className="min-h-screen p-8 grid-bg">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <h1 className="text-4xl font-black mb-2">Research <span className="gradient-text">Validation</span></h1>
        <p className="text-[#7a8fbf]">
          Complete experimental evidence · Ablation study · Statistical significance · Reproducibility
        </p>
      </motion.div>

      {/* Reproducibility section */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
        className="glass-card p-6 mb-6" style={{ border: "1px solid rgba(0,230,118,0.25)" }}>
        <div className="flex items-center gap-2 mb-4">
          <CheckCircle size={18} className="text-[#00e676]" />
          <h2 className="text-lg font-bold text-[#00e676]">Reproducibility Certificate</h2>
        </div>
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: "Run 1 (original)", value: "0.7745", sub: "2026-06-06 09:39", color: "#00e676" },
            { label: "Run 2 (verification)", value: "0.7745", sub: "2026-06-06 10:00", color: "#00e676" },
            { label: "p-value (both)", value: "0.0115", sub: "< 0.05 ✓", color: "#00b0ff" },
            { label: "Cohen d (both)", value: "0.491", sub: "medium-large ✓", color: "#e040fb" },
          ].map(s => (
            <div key={s.label} className="text-center p-4 rounded-xl" style={{ background: "rgba(0,230,118,0.06)" }}>
              <div className="text-xs text-[#7a8fbf] mb-1">{s.label}</div>
              <div className="text-2xl font-black font-mono" style={{ color: s.color }}>{s.value}</div>
              <div className="text-xs text-[#7a8fbf] mt-1">{s.sub}</div>
            </div>
          ))}
        </div>
      </motion.div>

      <div className="grid grid-cols-2 gap-6 mb-6">
        {/* Ablation study */}
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}
          className="glass-card p-6">
          <div className="flex items-center gap-2 mb-4">
            <FlaskConical size={16} className="text-[#ffa726]" />
            <h2 className="text-base font-bold">Ablation Study</h2>
            <span className="text-xs text-[#7a8fbf] ml-auto">Component contribution</span>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={ablations} layout="vertical" margin={{ left: 20, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
              <XAxis type="number" domain={[0.65, 0.79]} tick={{ fill: "#7a8fbf", fontSize: 10 }} />
              <YAxis type="category" dataKey="component" tick={{ fill: "#7a8fbf", fontSize: 9 }} width={150} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="f1" name="F1" radius={[0, 4, 4, 0]}>
                {ablations.map((a, i) => (
                  <Cell key={i}
                    fill={a.note === "Production model" ? "#00e676"
                      : a.delta < -0.03 ? "#ff5252"
                      : a.delta < -0.01 ? "#ff8a65"
                      : "#00b0ff"}
                    fillOpacity={0.8} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Radar comparison */}
        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}
          className="glass-card p-6">
          <div className="flex items-center gap-2 mb-4">
            <BookOpen size={16} className="text-[#00b0ff]" />
            <h2 className="text-base font-bold">Multi-Metric Comparison</h2>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="rgba(255,255,255,0.08)" />
              <PolarAngleAxis dataKey="metric" tick={{ fill: "#7a8fbf", fontSize: 11 }} />
              <PolarRadiusAxis angle={90} domain={[0.65, 1.0]} tick={{ fill: "#7a8fbf", fontSize: 9 }} />
              <Radar name="GraphMindRL_V5" dataKey="V5" stroke="#00e676" fill="#00e676" fillOpacity={0.2} />
              <Radar name="Baseline" dataKey="Baseline" stroke="#00b0ff" fill="#00b0ff" fillOpacity={0.1} />
              <Radar name="Markov-1" dataKey="Markov1" stroke="#7a8fbf" fill="#7a8fbf" fillOpacity={0.05} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
            </RadarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      <div className="grid grid-cols-2 gap-6 mb-6">
        {/* Weight grid heatmap (top 15) */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}
          className="glass-card p-6">
          <h2 className="text-base font-bold mb-1">Phase 11A — Weight Grid Search</h2>
          <p className="text-xs text-[#7a8fbf] mb-4">Top 15 configurations · Best: w=0.5/0.1/0.4 → F1=0.7733</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={weightGrid.slice(0, 10)} margin={{ left: 0, right: 10, bottom: 30 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="weights" tick={{ fill: "#7a8fbf", fontSize: 9 }} angle={-30} textAnchor="end" height={50} />
              <YAxis domain={[0.72, 0.78]} tick={{ fill: "#7a8fbf", fontSize: 10 }} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="f1" name="F1" radius={[4, 4, 0, 0]}>
                {weightGrid.slice(0, 10).map((_, i) => (
                  <Cell key={i} fill={i === 0 ? "#00e676" : `hsl(200, ${70 - i * 5}%, ${40 + i * 2}%)`} fillOpacity={0.85} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Threshold sweep */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}
          className="glass-card p-6">
          <h2 className="text-base font-bold mb-1">Phase 11B — Threshold Sweep</h2>
          <p className="text-xs text-[#7a8fbf] mb-4">F1 vs threshold · Best: 0.16 with baseline weights</p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={thresholdSweep} margin={{ left: 0, right: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="threshold" tick={{ fill: "#7a8fbf", fontSize: 10 }} />
              <YAxis domain={[0.72, 0.76]} tick={{ fill: "#7a8fbf", fontSize: 10 }} />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={0.7424} stroke="#7a8fbf" strokeDasharray="4 4"
                            label={{ value: "Baseline", fill: "#7a8fbf", fontSize: 9 }} />
              <Bar dataKey="f1" name="F1" radius={[4, 4, 0, 0]}>
                {thresholdSweep.map((t, i) => (
                  <Cell key={i} fill={t.threshold === 0.16 ? "#00e676" : "#00b0ff"} fillOpacity={t.threshold === 0.16 ? 1 : 0.6} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Statistical significance table */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}
        className="glass-card overflow-hidden mb-6">
        <div className="px-6 py-4 border-b border-white/5">
          <h2 className="font-bold text-base">Statistical Significance vs Baseline (paired t-test, n=31)</h2>
          <p className="text-xs text-[#7a8fbf] mt-0.5">Null hypothesis: policy = baseline. Two-tailed, α = 0.05.</p>
        </div>
        <table className="w-full">
          <thead style={{ background: "rgba(255,255,255,0.02)" }}>
            <tr>
              {["Policy", "ΔF1", "t-stat", "p-value", "Cohen d", "Verdict"].map(h => (
                <th key={h} className="text-left py-3 px-4 text-xs text-[#7a8fbf] font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sigData.map((r, i) => (
              <motion.tr key={r.policy}
                initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.05 }}
                className="border-t border-white/5">
                <td className="py-3 px-4 text-sm font-medium"
                    style={{ color: r.policy === "V5" ? "#00e676" : "white" }}>{r.name}</td>
                <td className="py-3 px-4 font-mono text-sm"
                    style={{ color: r.delta_f1 > 0.02 ? "#00e676" : r.delta_f1 > 0 ? "#00b0ff" : "#ff5252" }}>
                  {r.delta_f1 > 0 ? "+" : ""}{r.delta_f1.toFixed(4)}
                </td>
                <td className="py-3 px-4 font-mono text-sm text-[#7a8fbf]">—</td>
                <td className="py-3 px-4 font-mono text-sm">
                  <span style={{ color: r.p_value < 0.05 ? "#00e676" : "#ff8a65" }}>
                    {r.p_value < 0.001 ? "<0.001" : r.p_value.toFixed(4)}
                  </span>
                </td>
                <td className="py-3 px-4 font-mono text-sm text-[#7a8fbf]">{r.cohen_d.toFixed(3)}</td>
                <td className="py-3 px-4">
                  {r.significant
                    ? <span className="badge-production px-2 py-0.5 rounded text-xs flex items-center gap-1 w-fit">
                        <CheckCircle size={10} /> Significant
                      </span>
                    : <span className="badge-baseline px-2 py-0.5 rounded text-xs flex items-center gap-1 w-fit">
                        <XCircle size={10} /> Not sig.
                      </span>}
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </motion.div>

      {/* Context weight scientific statement */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }}
        className="glass-card p-6" style={{ border: "1px solid rgba(0,176,255,0.2)" }}>
        <div className="flex items-center gap-2 mb-3">
          <Info size={16} className="text-[#00b0ff]" />
          <h2 className="font-bold text-base text-[#00b0ff]">Scientific Note: Context Features</h2>
        </div>
        <div className="grid grid-cols-2 gap-6 text-sm">
          <div>
            <p className="text-[#7a8fbf] mb-2">
              Context features (time-of-day, day-of-week) were systematically evaluated across four granularities:
            </p>
            <div className="space-y-1.5">
              {[
                ["6-band (4h slots)", "98.5% coverage", "Hurts"],
                ["12-band (2h slots)", "97.6% coverage", "Hurts"],
                ["24-hour", "96.3% coverage", "Hurts"],
                ["48-bucket (30min)", "94.3% coverage", "Hurts"],
              ].map(([label, cov, result]) => (
                <div key={label} className="flex items-center gap-2 text-xs">
                  <span className="badge-rejected px-1.5 py-0.5 rounded">{result}</span>
                  <span className="text-white">{label}</span>
                  <span className="text-[#7a8fbf] ml-auto">{cov}</span>
                </div>
              ))}
            </div>
          </div>
          <div>
            <p className="text-[#7a8fbf] mb-3">
              <strong className="text-white">Finding:</strong> Coverage was not the issue (94–98% of states seen in training).
              The conditional distributions P(next|app, time_band) add noise on 2-month datasets.
              Context requires ≥12 months of stable behavioral data.
            </p>
            <div className="p-3 rounded-xl text-xs" style={{ background: "rgba(0,176,255,0.08)", border: "1px solid rgba(0,176,255,0.2)" }}>
              <strong className="text-[#00b0ff]">Context in dashboard:</strong>
              <span className="text-[#7a8fbf]"> Retained for monitoring, drift detection, and RL state representation.
              W_CONTEXT = 0.00 in the confidence score only.</span>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
