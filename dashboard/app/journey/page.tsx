"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import { CheckCircle2, XCircle, Minus } from "lucide-react";

interface Phase {
  phase: string; label: string; f1: number; delta: number;
  status: string; description: string; date: string;
}

const STATUS: Record<string, { label: string; badge: string; icon: any }> = {
  production: { label: "Production",    badge: "badge-green",  icon: CheckCircle2 },
  accepted:   { label: "Accepted",      badge: "badge-blue",   icon: CheckCircle2 },
  rejected:   { label: "Rejected",      badge: "badge-red",    icon: XCircle },
  failed:     { label: "Failed",        badge: "badge-red",    icon: XCircle },
  baseline:   { label: "Baseline",      badge: "badge-gray",   icon: Minus },
  weak:       { label: "Inconclusive",  badge: "badge-amber",  icon: Minus },
};

const Tip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as Phase;
  return (
    <div className="card-sm px-3 py-2 text-xs shadow-sm">
      <div className="font-medium text-gray-900">{d.label}</div>
      <div className="text-gray-500 mt-0.5">F1 = {d.f1.toFixed(4)}</div>
      <div style={{ color: d.delta >= 0 ? "#15803d" : "#b91c1c" }}>
        {d.delta >= 0 ? "+" : ""}{d.delta.toFixed(4)}
      </div>
    </div>
  );
};

export default function Journey() {
  const [phases, setPhases] = useState<Phase[]>([]);
  const [expanded, setExpanded] = useState<number | null>(null);

  useEffect(() => {
    fetch("/data/optimization.json").then(r => r.json()).then(setPhases).catch(() => {});
  }, []);

  return (
    <div className="max-w-4xl mx-auto px-8 py-10">
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
        <h1 className="page-title mb-1">Optimization Journey</h1>
        <p className="text-sm text-gray-500">
          8 phases · 5 hypotheses tested · From GraphOnly (0.7267) to GraphMindRL_V5 (0.7745)
        </p>
      </motion.div>

      {/* Summary stats */}
      <div className="grid grid-cols-4 gap-3 mb-6">
        {[
          { label: "Phases", value: "8" },
          { label: "Rejected", value: "3", red: true },
          { label: "Accepted", value: "3", green: true },
          { label: "ΔF1", value: "+0.0321", green: true },
        ].map(s => (
          <div key={s.label} className="card p-3.5">
            <div className="label mb-1.5">{s.label}</div>
            <div className={`text-xl font-semibold ${s.green ? "text-green-600" : s.red ? "text-red-500" : "text-gray-900"}`}>
              {s.value}
            </div>
          </div>
        ))}
      </div>

      {/* F1 trajectory */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}
        className="card p-5 mb-6">
        <h2 className="section-title mb-4">F1 Trajectory</h2>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={phases.map((p, i) => ({ ...p, i }))}>
            <CartesianGrid vertical={false} stroke="#f3f4f6" />
            <XAxis dataKey="label" tick={{ fill: "#9ca3af", fontSize: 10 }}
                   angle={-12} textAnchor="end" height={50} axisLine={false} tickLine={false} />
            <YAxis domain={[0.68, 0.79]} tick={{ fill: "#9ca3af", fontSize: 11 }}
                   axisLine={false} tickLine={false} tickFormatter={v => v.toFixed(3)} />
            <Tooltip content={<Tip />} />
            <ReferenceLine y={0.7424} stroke="#d1d5db" strokeDasharray="4 3"
                           label={{ value: "baseline", fill: "#9ca3af", fontSize: 10, position: "insideTopRight" }} />
            <ReferenceLine y={0.7745} stroke="#22c55e" strokeDasharray="4 3"
                           label={{ value: "V5", fill: "#16a34a", fontSize: 10, position: "insideTopRight" }} />
            <Line type="monotone" dataKey="f1" stroke="#111827" strokeWidth={1.5}
                  dot={(props) => {
                    const p = props.payload as Phase;
                    const color = p.status === "production" ? "#22c55e" : p.status === "accepted" ? "#3b82f6" : p.status === "rejected" || p.status === "failed" ? "#ef4444" : "#9ca3af";
                    return <circle key={props.index} cx={props.cx} cy={props.cy} r={4} fill={color} stroke="white" strokeWidth={1.5} />;
                  }} />
          </LineChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Phase timeline */}
      <div className="space-y-2">
        {phases.map((p, i) => {
          const cfg = STATUS[p.status] || STATUS.baseline;
          const Icon = cfg.icon;
          const open = expanded === i;
          return (
            <motion.div key={i} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.08 + i * 0.04 }}>
              <div className={`card cursor-pointer transition-colors ${open ? "" : "hover:border-gray-300"}`}
                   onClick={() => setExpanded(open ? null : i)}>
                <div className="px-5 py-4 flex items-center gap-4">
                  {/* Icon */}
                  <div className={`w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 ${
                    p.status === "production" ? "bg-green-50" :
                    p.status === "accepted" ? "bg-blue-50" :
                    p.status === "failed" || p.status === "rejected" ? "bg-red-50" : "bg-gray-50"
                  }`}>
                    <Icon size={13} className={
                      p.status === "production" ? "text-green-600" :
                      p.status === "accepted" ? "text-blue-600" :
                      p.status === "failed" || p.status === "rejected" ? "text-red-500" : "text-gray-400"
                    } />
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className="text-xs text-gray-400">{p.phase}</span>
                      <span className={`badge ${cfg.badge}`}>{cfg.label}</span>
                    </div>
                    <div className="text-sm font-medium text-gray-900 mt-0.5">{p.label}</div>
                    {!open && <div className="text-xs text-gray-500 mt-0.5 truncate">{p.description}</div>}
                  </div>

                  {/* F1 */}
                  <div className="text-right flex-shrink-0">
                    <div className="mono text-base font-semibold text-gray-900">{p.f1.toFixed(4)}</div>
                    <div className={`mono text-xs ${p.delta >= 0 ? "text-green-600" : "text-red-500"}`}>
                      {p.delta >= 0 ? "+" : ""}{p.delta.toFixed(4)}
                    </div>
                  </div>
                </div>

                {open && (
                  <div className="px-5 pb-4 pt-0" style={{ borderTop: "1px solid #f3f4f6" }}>
                    <p className="text-sm text-gray-600 mt-3">{p.description}</p>
                    <div className="flex gap-4 mt-3">
                      <div className="text-xs"><span className="text-gray-400">Date </span><span className="text-gray-700">{p.date}</span></div>
                      <div className="text-xs"><span className="text-gray-400">F1 </span><span className="mono font-medium text-gray-900">{p.f1.toFixed(4)}</span></div>
                      <div className="text-xs"><span className="text-gray-400">Delta </span>
                        <span className={`mono font-medium ${p.delta >= 0 ? "text-green-600" : "text-red-500"}`}>
                          {p.delta >= 0 ? "+" : ""}{p.delta.toFixed(4)}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
