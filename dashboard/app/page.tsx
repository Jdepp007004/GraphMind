"use client";
import { motion } from "framer-motion";
import Link from "next/link";
import { ArrowRight, TrendingUp, Shield, Zap, Database, Target, Layers } from "lucide-react";

const stats = [
  { label: "Cache Hit Rate", value: "97.92%", sub: "Real UbiqLog · 31 users", icon: Target, color: "#22c55e", dot: "dot-green" },
  { label: "Memory Thrash", value: "0.00%", sub: "↓ from 33.98% (LRU)", icon: Shield, color: "#3b82f6", dot: "dot-blue" },
  { label: "Latency Saved", value: "385ms", sub: "avg per app launch", icon: Zap, color: "#f59e0b", dot: "dot-amber" },
  { label: "Cache Tiers", value: "5", sub: "PIN · HOT · WARM · COOL · COLD", icon: Layers, color: "#8b5cf6", dot: "dot-blue" },
  { label: "KPIs Passing", value: "7 / 7", sub: "All PS03 targets met", icon: TrendingUp, color: "#22c55e", dot: "dot-green" },
  { label: "Users Evaluated", value: "31", sub: "UbiqLog · real Android data", icon: Database, color: "#6b7280", dot: "dot-gray" },
];

const pipeline = [
  { step: "Dataset", desc: "UbiqLog4UCI · 31 users · 508 days · real Android app-switch events" },
  { step: "Behavioural Graph", desc: "Per-user weighted Markov graph · P(next | current app)" },
  { step: "Confidence Scorer", desc: "0.50 × P_trans + 0.40 × Frequency + 0.10 × Recency" },
  { step: "Transformer Reranker", desc: "Per-user EmbeddingTransformerReranker · 34-dim app embeddings" },
  { step: "RL Controller (PPO)", desc: "Adaptive threshold ±0.005 · 20-step rolling hit rate" },
  { step: "5-Tier Cache", desc: "PIN (10ms) → HOT (42ms) → WARM (190ms) → COOL (400ms) → COLD (720ms)" },
];

const config = [
  { k: "PIN_TIER_CAPACITY",  v: "3",   note: "Always in RAM" },
  { k: "HOT_TIER_CAPACITY",  v: "5",   note: "LRU active" },
  { k: "WARM_TIER_CAPACITY", v: "8",   note: "AI prefetched" },
  { k: "COOL_TIER_CAPACITY", v: "20",  note: "← V6 innovation" },
  { k: "W_TRANSITION",       v: "0.50", note: "" },
  { k: "W_FREQUENCY",        v: "0.40", note: "" },
  { k: "W_RECENCY",          v: "0.10", note: "" },
];

const quickLinks = [
  { href: "/benchmark",    label: "Benchmark Explorer",  desc: "14-policy comparison · V6 vs all baselines" },
  { href: "/kpi",          label: "KPI Dashboard",        desc: "7/7 PS03 KPIs · real numbers" },
  { href: "/architecture", label: "V6 Architecture",      desc: "5-tier memory & transformer reranker design" },
];

const kpiRows = [
  { kpi: "Cache Hit Rate",             target: "≥ 85%",  achieved: "97.92%",  pass: true },
  { kpi: "Memory Thrashing Reduction", target: "≥ 50%",  achieved: "100.00%", pass: true },
  { kpi: "App Load Time Improvement",  target: "≥ 20%",  achieved: "72.18%",  pass: true },
  { kpi: "App Launch Time Improvement",target: "≥ 10%",  achieved: "82.20%",  pass: true },
  { kpi: "Context Prediction (F1)",    target: "≥ 75%",  achieved: "97.92%",  pass: true },
  { kpi: "System Stability",           target: "0 issues","achieved": "0",     pass: true },
  { kpi: "Memory Efficiency vs LRU",   target: "≥ 30%",  achieved: "96.91%",  pass: true },
];

export default function Overview() {
  return (
    <div className="max-w-5xl mx-auto px-8 py-10">

      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <div className="flex items-center gap-2 text-xs text-gray-400 mb-3">
          <span>Samsung EnnovateX AX 2026</span>
          <span>·</span>
          <span>PS03 — Context-Aware Memory</span>
        </div>
        <h1 className="text-2xl font-semibold text-gray-900 mb-1.5">GraphMind V6</h1>
        <p className="text-sm text-gray-500 max-w-xl">
          Per-user Transformer reranking on Markov graphs with a 5-tier memory hierarchy.
          Benchmarked on 31 real smartphone users from the UbiqLog dataset — 7/7 PS03 KPIs passing.
        </p>
        <div className="flex items-center gap-2 mt-4">
          <span className="badge badge-green">7/7 KPIs PASS</span>
          <span className="badge badge-blue">5-Tier Cache</span>
          <span className="badge badge-gray">Samsung Galaxy A23 · Real latency</span>
          <span className="badge badge-amber">Transformer Reranker</span>
        </div>
      </motion.div>

      {/* Stats grid */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}
        className="grid grid-cols-3 gap-3 mb-8">
        {stats.map((s) => (
          <div key={s.label} className="card p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="label">{s.label}</span>
              <s.icon size={14} className="text-gray-300" />
            </div>
            <div className="stat-value" style={{ color: s.color }}>{s.value}</div>
            <div className="text-xs text-gray-400 mt-1">{s.sub}</div>
          </div>
        ))}
      </motion.div>

      <div className="grid grid-cols-5 gap-5 mb-8">
        {/* Pipeline */}
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
          className="col-span-3 card p-5">
          <h2 className="section-title mb-4">V6 System Pipeline</h2>
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
          <h2 className="section-title mb-4">V6 Configuration</h2>
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
            <div className="text-xs text-gray-400 mb-1">Benchmark result (UbiqLog real data)</div>
            <div className="font-semibold text-gray-900">Cache Hit Rate = 97.92% · 7/7 KPIs PASS</div>
          </div>
        </motion.div>
      </div>

      {/* KPI Table */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.13 }}
        className="card overflow-hidden mb-8">
        <div className="px-5 py-3.5" style={{ borderBottom: "1px solid #f3f4f6" }}>
          <h2 className="section-title">PS03 KPI Results</h2>
        </div>
        <table className="w-full">
          <thead style={{ background: "#fafafa", borderBottom: "1px solid #f3f4f6" }}>
            <tr>
              <th className="text-left py-2.5 px-4 label">KPI</th>
              <th className="text-left py-2.5 px-4 label">Target</th>
              <th className="text-left py-2.5 px-4 label">Achieved</th>
              <th className="text-left py-2.5 px-4 label">Status</th>
            </tr>
          </thead>
          <tbody>
            {kpiRows.map(r => (
              <tr key={r.kpi} className="table-row">
                <td className="py-2.5 px-4 text-sm text-gray-700">{r.kpi}</td>
                <td className="py-2.5 px-4 mono text-sm text-gray-500">{r.target}</td>
                <td className="py-2.5 px-4 mono text-sm font-semibold text-gray-900">{r.achieved}</td>
                <td className="py-2.5 px-4">
                  {r.pass
                    ? <span className="badge badge-green">✓ PASS</span>
                    : <span className="badge badge-red">✗ FAIL</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </motion.div>

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
