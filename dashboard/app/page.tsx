"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  BarChart2, Users, Zap, TrendingUp, Database, GitMerge,
  ChevronRight, Shield, Brain, ArrowRight,
} from "lucide-react";
import Link from "next/link";

interface Summary {
  f1: number; delta_f1: number; baseline_f1: number;
  hit_rate: number; latency_saved_ms: number;
  p_value: number; cohen_d: number; n_users: number;
  n_transitions: number; n_events: number;
  cold_start_ms: number; device: string;
  config: { w_transition: number; w_recency: number; w_frequency: number; threshold: number; hot_size: number; warm_size: number; };
}

function useData<T>(url: string, fallback: T): T {
  const [data, setData] = useState<T>(fallback);
  useEffect(() => {
    fetch(url).then(r => r.json()).then(setData).catch(() => {});
  }, [url]);
  return data;
}

function AnimatedNumber({ value, decimals = 0, prefix = "", suffix = "" }: { value: number; decimals?: number; prefix?: string; suffix?: string }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    let start = 0;
    const end = value;
    const dur = 1500;
    const step = 16;
    const steps = dur / step;
    const increment = (end - start) / steps;
    let current = start;
    const timer = setInterval(() => {
      current += increment;
      if (current >= end) { setDisplay(end); clearInterval(timer); }
      else setDisplay(current);
    }, step);
    return () => clearInterval(timer);
  }, [value]);
  return <span>{prefix}{display.toFixed(decimals)}{suffix}</span>;
}

const heroMetrics = [
  { label: "F1 Score", value: 0.7745, decimals: 4, icon: TrendingUp, color: "#00e676", glow: "rgba(0,230,118,0.2)", suffix: "", desc: "GraphMindRL_V5" },
  { label: "Improvement", value: 3.21, decimals: 2, icon: BarChart2, color: "#00b0ff", glow: "rgba(0,176,255,0.2)", suffix: "%", desc: "vs baseline" },
  { label: "Hit Rate", value: 93.07, decimals: 1, icon: Zap, color: "#ffa726", glow: "rgba(255,167,38,0.2)", suffix: "%", desc: "cache hit rate" },
  { label: "Latency Saved", value: 1847, decimals: 0, icon: Brain, color: "#e040fb", glow: "rgba(224,64,251,0.2)", suffix: "ms", desc: "per launch avg" },
  { label: "Users", value: 31, decimals: 0, icon: Users, color: "#00e5ff", glow: "rgba(0,229,255,0.2)", suffix: "", desc: "UbiqLog dataset" },
  { label: "Transitions", value: 208695, decimals: 0, icon: Database, color: "#69f0ae", glow: "rgba(105,240,174,0.2)", suffix: "", desc: "reconstructed" },
];

const architectureLayers = [
  { label: "UbiqLog Dataset", desc: "35 users · 9.7M events · 2 months", color: "#7a8fbf" },
  { label: "Transition Pipeline", desc: "MAX_GAP=3600s · 208,695 transitions", color: "#00b0ff" },
  { label: "Markov Graph", desc: "Per-user weighted directed graph", color: "#1428a0" },
  { label: "Confidence Scorer", desc: "0.5×P + 0.1×Recency + 0.4×Frequency", color: "#00e676" },
  { label: "RL Threshold Controller", desc: "Adaptive ±0.005 · 20-step hit rate", color: "#ffa726" },
  { label: "Prefetch Cache", desc: "HOT=5 · WARM=15 · COLD=SQLite", color: "#e040fb" },
];

export default function ExecutiveOverview() {
  const summary = useData<Summary>("/data/summary.json", {} as Summary);

  return (
    <div className="min-h-screen p-8 grid-bg">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-10">
        <div className="flex items-center gap-2 text-xs text-[#7a8fbf] mb-3">
          <span>Samsung EnnovateX AX Hackathon 2025</span>
          <ChevronRight size={12} />
          <span className="text-[#00b0ff]">Research Submission</span>
        </div>
        <h1 className="text-5xl font-black mb-2 tracking-tight">
          <span className="gradient-text">GraphMind</span>
          <span className="text-white">RL</span>
          <span className="text-[#00b0ff] text-3xl font-bold ml-2 align-middle">V5</span>
        </h1>
        <p className="text-[#7a8fbf] text-lg max-w-2xl">
          Reinforcement Learning on Markov graphs for intelligent Android app prefetching.
          Statistically validated on real UbiqLog smartphone dataset.
        </p>
        <div className="flex items-center gap-3 mt-4">
          <span className="badge-production px-3 py-1 rounded-full text-xs font-semibold">
            ✓ p = 0.0115 &lt; 0.05
          </span>
          <span className="badge-production px-3 py-1 rounded-full text-xs font-semibold">
            ✓ Cohen d = 0.491
          </span>
          <span className="badge-accepted px-3 py-1 rounded-full text-xs font-semibold">
            Samsung Galaxy A23 · Real Latency
          </span>
        </div>
      </motion.div>

      {/* Hero Metrics Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        {heroMetrics.map((m, i) => (
          <motion.div key={m.label}
            initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.08 }}
            className="glass-card p-5 relative overflow-hidden"
            style={{ boxShadow: `0 0 30px ${m.glow}` }}>
            <div className="absolute top-0 right-0 w-32 h-32 rounded-full opacity-10"
                 style={{ background: m.glow, transform: "translate(30%,-30%)" }} />
            <div className="flex items-start justify-between mb-3">
              <m.icon size={20} style={{ color: m.color }} />
              <span className="text-[10px] text-[#7a8fbf] uppercase tracking-wider">{m.desc}</span>
            </div>
            <div className="text-3xl font-black metric-number" style={{ color: m.color }}>
              <AnimatedNumber value={m.value} decimals={m.decimals} suffix={m.suffix} />
            </div>
            <div className="text-sm text-[#7a8fbf] mt-1">{m.label}</div>
          </motion.div>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-6 mb-8">
        {/* Architecture Diagram */}
        <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 }}
          className="col-span-2 glass-card p-6">
          <h2 className="text-lg font-bold mb-5 flex items-center gap-2">
            <GitMerge size={18} className="text-[#00b0ff]" />
            System Architecture
          </h2>
          <div className="space-y-2">
            {architectureLayers.map((layer, i) => (
              <motion.div key={layer.label}
                initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 + i * 0.07 }}
                className="flex items-center gap-4">
                <div className="flex flex-col items-center">
                  <div className="w-3 h-3 rounded-full border-2" style={{ borderColor: layer.color, background: layer.color + "33" }} />
                  {i < architectureLayers.length - 1 && (
                    <div className="w-0.5 h-6 my-1" style={{ background: `linear-gradient(${layer.color}, ${architectureLayers[i+1].color})`, opacity: 0.4 }} />
                  )}
                </div>
                <div className="flex-1 py-1 px-4 rounded-lg transition-all"
                     style={{ background: `linear-gradient(135deg, ${layer.color}12, transparent)`, border: `1px solid ${layer.color}25` }}>
                  <div className="font-semibold text-sm" style={{ color: layer.color }}>{layer.label}</div>
                  <div className="text-xs text-[#7a8fbf]">{layer.desc}</div>
                </div>
                {i < architectureLayers.length - 1 && (
                  <ArrowRight size={12} style={{ color: layer.color, opacity: 0.5 }} />
                )}
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Production Config */}
        <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.4 }}
          className="glass-card p-6 flex flex-col">
          <h2 className="text-lg font-bold mb-5 flex items-center gap-2">
            <Shield size={18} className="text-[#00e676]" />
            Production Config
          </h2>
          <div className="space-y-3 flex-1">
            {[
              { label: "W_TRANSITION", value: "0.50", color: "#00b0ff" },
              { label: "W_RECENCY",    value: "0.10 ↓", color: "#ffa726", note: "was 0.30" },
              { label: "W_FREQUENCY",  value: "0.40 ↑", color: "#00e676", note: "was 0.20" },
              { label: "W_CONTEXT",    value: "0.00",   color: "#7a8fbf", note: "zeroed" },
              { label: "THRESHOLD",    value: "0.16 ±Δ", color: "#e040fb" },
              { label: "HOT_SIZE",     value: "5",      color: "#ffa726" },
              { label: "WARM_SIZE",    value: "15",     color: "#00b0ff" },
            ].map(item => (
              <div key={item.label} className="flex items-center justify-between py-1.5 px-3 rounded-lg"
                   style={{ background: "rgba(255,255,255,0.03)" }}>
                <div>
                  <span className="text-xs font-mono text-[#7a8fbf]">{item.label}</span>
                  {item.note && <span className="text-xs text-[#7a8fbf]/50 ml-2">({item.note})</span>}
                </div>
                <span className="font-mono text-sm font-bold" style={{ color: item.color }}>{item.value}</span>
              </div>
            ))}
          </div>
          <div className="mt-4 p-3 rounded-xl badge-production text-center">
            <div className="text-xs font-semibold">F1 = 0.7745</div>
            <div className="text-xs opacity-70">Reproduced × 2 ✓</div>
          </div>
        </motion.div>
      </div>

      {/* Quick nav cards */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.7 }}>
        <h2 className="text-lg font-bold mb-4 text-[#7a8fbf]">Explore the Dashboard</h2>
        <div className="grid grid-cols-3 gap-3">
          {[
            { href: "/benchmark",  label: "Benchmark Explorer",   desc: "F1 · Hit Rate · Latency · Stats",   color: "#00b0ff" },
            { href: "/journey",    label: "Optimization Journey",  desc: "8 phases · 5 hypotheses tested",     color: "#00e676" },
            { href: "/graph",      label: "Graph Explorer",        desc: "App transitions · React Flow",        color: "#e040fb" },
            { href: "/simulator",  label: "Cache Simulator",       desc: "HOT/WARM live simulation",            color: "#ffa726" },
            { href: "/playback",   label: "User Playback",         desc: "Event timeline · Confidence scores",  color: "#00e5ff" },
            { href: "/research",   label: "Research Validation",   desc: "Significance · Ablations · Repro",    color: "#69f0ae" },
          ].map(card => (
            <Link key={card.href} href={card.href}
              className="glass-card p-4 hover:scale-[1.02] transition-transform cursor-pointer group"
              style={{ borderColor: `${card.color}20` }}>
              <div className="text-sm font-semibold mb-1 group-hover:text-white transition-colors"
                   style={{ color: card.color }}>{card.label}</div>
              <div className="text-xs text-[#7a8fbf]">{card.desc}</div>
            </Link>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
