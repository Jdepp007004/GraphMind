"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { CheckCircle, XCircle, Clock, Lightbulb } from "lucide-react";

interface Phase {
  phase: string; label: string; f1: number; delta: number;
  status: string; description: string; date: string; result: string;
}

const STATUS_CONFIG: Record<string, { label: string; color: string; icon: any; cls: string }> = {
  production: { label: "PRODUCTION",  color: "#00e676", icon: CheckCircle, cls: "badge-production" },
  accepted:   { label: "ACCEPTED",    color: "#00b0ff", icon: CheckCircle, cls: "badge-accepted" },
  rejected:   { label: "REJECTED",    color: "#ff5252", icon: XCircle,     cls: "badge-rejected" },
  failed:     { label: "FAILED",      color: "#ff5252", icon: XCircle,     cls: "badge-rejected" },
  baseline:   { label: "BASELINE",    color: "#7a8fbf", icon: Clock,       cls: "badge-baseline" },
  weak:       { label: "INCONCLUSIVE",color: "#ffa726", icon: Lightbulb,   cls: "badge-weak" },
};

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload as Phase;
  return (
    <div className="glass-card p-4 max-w-xs text-xs">
      <p className="font-bold text-white mb-1">{d.label}</p>
      <p className="text-[#7a8fbf] mb-2">{d.phase}</p>
      <div className="flex gap-4 mb-2">
        <span>F1: <strong className="text-white">{d.f1.toFixed(4)}</strong></span>
        <span style={{ color: d.delta >= 0 ? "#00e676" : "#ff5252" }}>
          {d.delta >= 0 ? "+" : ""}{d.delta.toFixed(4)}
        </span>
      </div>
    </div>
  );
};

export default function OptimizationJourney() {
  const [phases, setPhases] = useState<Phase[]>([]);
  const [active, setActive] = useState<number | null>(null);

  useEffect(() => {
    fetch("/data/optimization.json").then(r => r.json()).then(setPhases).catch(() => {});
  }, []);

  const lineData = phases.map((p, i) => ({ ...p, index: i }));
  const baseline = 0.7424;

  return (
    <div className="min-h-screen p-8 grid-bg">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <h1 className="text-4xl font-black mb-2">Optimization <span className="gradient-text">Journey</span></h1>
        <p className="text-[#7a8fbf]">
          Systematic hypothesis testing · 8 phases · 5 approaches evaluated ·
          From GraphOnly to GraphMindRL_V5 (+0.0321 F1)
        </p>
      </motion.div>

      {/* F1 trajectory chart */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
        className="glass-card p-6 mb-8">
        <h2 className="text-lg font-bold mb-4 text-white">F1 Score Trajectory</h2>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={lineData} margin={{ left: 0, right: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="label" tick={{ fill: "#7a8fbf", fontSize: 10 }}
                   angle={-15} textAnchor="end" height={60} />
            <YAxis domain={[0.68, 0.79]} tick={{ fill: "#7a8fbf", fontSize: 11 }}
                   tickFormatter={v => v.toFixed(3)} />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={baseline} stroke="#7a8fbf" strokeDasharray="4 4"
                           label={{ value: "Baseline 0.7424", fill: "#7a8fbf", fontSize: 10, position: "insideTopRight" }} />
            <ReferenceLine y={0.7745} stroke="#00e676" strokeDasharray="4 4"
                           label={{ value: "V5 0.7745", fill: "#00e676", fontSize: 10, position: "insideTopRight" }} />
            <Line type="monotone" dataKey="f1" stroke="#00b0ff" strokeWidth={2}
                  dot={(props) => {
                    const p = props.payload as Phase;
                    const color = STATUS_CONFIG[p.status]?.color || "#00b0ff";
                    return <circle key={props.index} cx={props.cx} cy={props.cy} r={6}
                                   fill={color} stroke="#050917" strokeWidth={2} />;
                  }}
                  activeDot={{ r: 8, fill: "#00b0ff" }} />
          </LineChart>
        </ResponsiveContainer>
      </motion.div>

      {/* Timeline */}
      <div className="space-y-4">
        {phases.map((phase, i) => {
          const cfg = STATUS_CONFIG[phase.status] || STATUS_CONFIG.baseline;
          const Icon = cfg.icon;
          const isActive = active === i;
          return (
            <motion.div key={i}
              initial={{ opacity: 0, x: -30 }} animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 + i * 0.06 }}
              className="flex gap-4 cursor-pointer"
              onClick={() => setActive(isActive ? null : i)}>
              
              {/* Timeline connector */}
              <div className="flex flex-col items-center flex-shrink-0">
                <div className="w-10 h-10 rounded-full flex items-center justify-center border-2 transition-all"
                     style={{ borderColor: cfg.color, background: `${cfg.color}18`,
                              boxShadow: isActive ? `0 0 20px ${cfg.color}44` : "none" }}>
                  <Icon size={18} style={{ color: cfg.color }} />
                </div>
                {i < phases.length - 1 && (
                  <div className="w-0.5 flex-1 mt-2" style={{ background: `linear-gradient(${cfg.color}60, ${STATUS_CONFIG[phases[i+1].status]?.color || "#00b0ff"}30)` }} />
                )}
              </div>

              {/* Card */}
              <div className={`flex-1 glass-card p-5 mb-4 transition-all duration-300 ${isActive ? "glow-cyan" : ""}`}
                   style={{ borderColor: isActive ? `${cfg.color}40` : "rgba(255,255,255,0.06)" }}>
                <div className="flex items-start justify-between flex-wrap gap-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 flex-wrap mb-1">
                      <span className="text-xs text-[#7a8fbf]">{phase.phase}</span>
                      <span className={`${cfg.cls} px-2 py-0.5 rounded text-xs font-semibold`}>{cfg.label}</span>
                    </div>
                    <h3 className="text-base font-bold text-white mb-2">{phase.label}</h3>
                    <p className="text-sm text-[#7a8fbf]">{phase.description}</p>
                  </div>
                  <div className="text-right flex-shrink-0">
                    <div className="font-mono text-2xl font-black" style={{ color: cfg.color }}>
                      {phase.f1.toFixed(4)}
                    </div>
                    <div className={`font-mono text-sm ${phase.delta >= 0 ? "text-[#00e676]" : "text-[#ff5252]"}`}>
                      {phase.delta >= 0 ? "+" : ""}{phase.delta.toFixed(4)}
                    </div>
                    <div className="text-xs text-[#7a8fbf]">{phase.date}</div>
                  </div>
                </div>

                {isActive && phase.status !== "baseline" && (
                  <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }}
                    className="mt-4 pt-4 border-t border-white/10">
                    <div className="grid grid-cols-3 gap-3 text-center">
                      <div className="p-3 rounded-xl" style={{ background: "rgba(255,255,255,0.04)" }}>
                        <div className="text-xs text-[#7a8fbf] mb-1">F1 Achieved</div>
                        <div className="font-mono font-bold" style={{ color: cfg.color }}>{phase.f1.toFixed(4)}</div>
                      </div>
                      <div className="p-3 rounded-xl" style={{ background: "rgba(255,255,255,0.04)" }}>
                        <div className="text-xs text-[#7a8fbf] mb-1">Delta vs Baseline</div>
                        <div className={`font-mono font-bold ${phase.delta >= 0 ? "text-[#00e676]" : "text-[#ff5252]"}`}>
                          {phase.delta >= 0 ? "+" : ""}{phase.delta.toFixed(4)}
                        </div>
                      </div>
                      <div className="p-3 rounded-xl" style={{ background: "rgba(255,255,255,0.04)" }}>
                        <div className="text-xs text-[#7a8fbf] mb-1">Decision</div>
                        <div className={`text-xs font-semibold ${cfg.cls} px-2 py-1 rounded`}>{cfg.label}</div>
                      </div>
                    </div>
                  </motion.div>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Summary */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.0 }}
        className="glass-card p-6 mt-6 glow-green"
        style={{ border: "1px solid rgba(0,230,118,0.3)" }}>
        <h2 className="text-lg font-bold text-[#00e676] mb-4">Research Methodology</h2>
        <div className="grid grid-cols-4 gap-4 text-center">
          {[
            { label: "Phases Completed", value: "8", color: "#00b0ff" },
            { label: "Hypotheses Rejected", value: "3", color: "#ff5252" },
            { label: "Improvements Accepted", value: "3", color: "#00e676" },
            { label: "F1 Improvement", value: "+0.0321", color: "#00e676" },
          ].map(s => (
            <div key={s.label}>
              <div className="text-3xl font-black metric-number mb-1" style={{ color: s.color }}>{s.value}</div>
              <div className="text-xs text-[#7a8fbf]">{s.label}</div>
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
