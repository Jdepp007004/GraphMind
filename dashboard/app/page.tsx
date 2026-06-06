"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { ArrowRight, TrendingUp, Users, Zap, Database, Target, Clock } from "lucide-react";

interface Summary {
  f1: number; delta_f1: number; baseline_f1: number;
  hit_rate: number; latency_saved_ms: number;
  p_value: number; cohen_d: number; n_users: number;
  n_transitions: number; n_events: number;
  config: { w_transition: number; w_recency: number; w_frequency: number; threshold: number; hot_size: number; warm_size: number; };
}

function useData<T>(url: string, fallback: T): T {
  const [data, setData] = useState<T>(fallback);
  useEffect(() => {
    fetch(url).then(r => r.json()).then(setData).catch(() => {});
  }, [url]);
  return data;
}

const stats = [
  { label: "F1 Score", value: "0.7745", sub: "+0.0321 vs baseline", icon: TrendingUp, color: "#22c55e", dot: "dot-green" },
  { label: "Cache Hit Rate", value: "93.1%", sub: "of test events", icon: Target, color: "#3b82f6", dot: "dot-blue" },
  { label: "Latency Saved", value: "1,847ms", sub: "per app launch avg", icon: Zap, color: "#f59e0b", dot: "dot-amber" },
  { label: "Users", value: "31", sub: "UbiqLog dataset", icon: Users, color: "#6b7280", dot: "dot-gray" },
  { label: "Transitions", value: "208,695", sub: "reconstructed", icon: Database, color: "#6b7280", dot: "dot-gray" },
  { label: "p-value", value: "0.0115", sub: "Cohen d = 0.491", icon: Clock, color: "#6b7280", dot: "dot-gray" },
];

const pipeline = [
  { step: "Dataset", desc: "UbiqLog4UCI · 35 users · 9.7M events · ~2 months" },
  { step: "Transitions", desc: "MAX_GAP = 3600s · 208,695 valid transitions extracted" },
  { step: "Markov Graph", desc: "Per-user weighted directed graph · P(next | current)" },
  { step: "Confidence Score", desc: "0.5 × P_trans + 0.1 × Recency + 0.4 × Frequency" },
  { step: "RL Controller", desc: "Adaptive threshold ±0.005 · 20-step rolling hit rate" },
  { step: "Prefetch Cache", desc: "HOT = 5 apps · WARM = 15 apps · COLD = SQLite" },
];

const config = [
  { k: "W_TRANSITION", v: "0.50", note: "" },
  { k: "W_RECENCY",    v: "0.10", note: "↓ was 0.30" },
  { k: "W_FREQUENCY",  v: "0.40", note: "↑ was 0.20" },
  { k: "W_CONTEXT",    v: "0.00", note: "zeroed · noisy" },
  { k: "THRESHOLD",    v: "0.16", note: "adaptive ±0.005" },
  { k: "HOT_SIZE",     v: "5",   note: "" },
  { k: "WARM_SIZE",    v: "15",  note: "" },
];

const quickLinks = [
  { href: "/benchmark",  label: "Benchmark Explorer",  desc: "Full policy comparison" },
  { href: "/journey",    label: "Optimization Journey", desc: "8 phases, 5 hypotheses" },
  { href: "/graph",      label: "Graph Explorer",       desc: "Interactive Markov graph" },
  { href: "/simulator",  label: "Cache Simulator",      desc: "Live HOT/WARM simulation" },
  { href: "/playback",   label: "User Playback",        desc: "Step through real events" },
  { href: "/research",   label: "Research Validation",  desc: "Stats, ablations, repro" },
];

export default function Overview() {
  const summary = useData<Partial<Summary>>("/data/summary.json", {});

  return (
    <div className="max-w-5xl mx-auto px-8 py-10">

      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <div className="flex items-center gap-2 text-xs text-gray-400 mb-3">
          <span>Samsung EnnovateX AX 2025</span>
          <span>·</span>
          <span>Research Submission</span>
        </div>
        <h1 className="text-2xl font-semibold text-gray-900 mb-1.5">GraphMindRL V5</h1>
        <p className="text-sm text-gray-500 max-w-xl">
          Reinforcement learning on Markov graphs for intelligent Android app prefetching.
          Statistically validated on the UbiqLog dataset across 31 real smartphone users.
        </p>
        <div className="flex items-center gap-2 mt-4">
          <span className="badge badge-green">p = 0.0115 &lt; 0.05</span>
          <span className="badge badge-blue">Cohen d = 0.491</span>
          <span className="badge badge-gray">Samsung Galaxy A23 · Real latency</span>
        </div>
      </motion.div>

      {/* Stats grid */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}
        className="grid grid-cols-3 gap-3 mb-8">
        {stats.map((s, i) => (
          <div key={s.label} className="card p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="label">{s.label}</span>
              <s.icon size={14} className="text-gray-300" />
            </div>
            <div className="stat-value text-gray-900">{s.value}</div>
            <div className="text-xs text-gray-400 mt-1">{s.sub}</div>
          </div>
        ))}
      </motion.div>

      <div className="grid grid-cols-5 gap-5 mb-8">
        {/* Pipeline */}
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
          className="col-span-3 card p-5">
          <h2 className="section-title mb-4">System Pipeline</h2>
          <div className="space-y-0">
            {pipeline.map((p, i) => (
              <div key={p.step} className="flex gap-3">
                <div className="flex flex-col items-center">
                  <div className="w-5 h-5 rounded-full bg-gray-100 border border-gray-200 flex items-center justify-center flex-shrink-0 mt-0.5">
                    <span className="text-xs font-semibold text-gray-500">{i + 1}</span>
                  </div>
                  {i < pipeline.length - 1 && (
                    <div className="w-px flex-1 bg-gray-100 my-1" />
                  )}
                </div>
                <div className="pb-4">
                  <div className="text-sm font-medium text-gray-900">{p.step}</div>
                  <div className="text-xs text-gray-500 mt-0.5">{p.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Config */}
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.12 }}
          className="col-span-2 card p-5 flex flex-col">
          <h2 className="section-title mb-4">Production Config</h2>
          <div className="space-y-1 flex-1">
            {config.map(c => (
              <div key={c.k} className="flex items-center justify-between py-1.5"
                   style={{ borderBottom: "1px solid #f9fafb" }}>
                <div>
                  <span className="mono text-xs text-gray-600">{c.k}</span>
                  {c.note && <span className="text-xs text-gray-400 ml-2">{c.note}</span>}
                </div>
                <span className="mono text-sm font-semibold text-gray-900">{c.v}</span>
              </div>
            ))}
          </div>
          <div className="mt-4 pt-4" style={{ borderTop: "1px solid #f3f4f6" }}>
            <div className="text-xs text-gray-400 mb-1">Benchmark result</div>
            <div className="font-semibold text-gray-900">F1 = 0.7745 · Reproduced ×2</div>
          </div>
        </motion.div>
      </div>

      {/* Quick nav */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
        <h2 className="section-title mb-3">Dashboard</h2>
        <div className="grid grid-cols-3 gap-2">
          {quickLinks.map(l => (
            <Link key={l.href} href={l.href}>
              <div className="card p-3.5 flex items-center justify-between group hover:border-gray-300 transition-colors cursor-pointer">
                <div>
                  <div className="text-sm font-medium text-gray-900">{l.label}</div>
                  <div className="text-xs text-gray-400 mt-0.5">{l.desc}</div>
                </div>
                <ArrowRight size={14} className="text-gray-300 group-hover:text-gray-500 transition-colors" />
              </div>
            </Link>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
