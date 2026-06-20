"use client";
import { useState } from "react";
import { motion } from "framer-motion";
import {
  Layers, Cpu, Shield, Zap, Sparkles, Database,
  Key, Info, CheckCircle2, Server
} from "lucide-react";

interface TierInfo {
  name: string;
  latency: string;
  capacity: string;
  role: string;
  description: string;
  color: string;
  bg: string;
}

const TIERS: TierInfo[] = [
  {
    name: "PIN Tier",
    latency: "10ms",
    capacity: "3 Apps",
    role: "System Critical Apps",
    description: "Kept permanently in active RAM. Holds core applications (e.g., Dialer, Messages, System Settings) to guarantee instant startup and eliminate cold launch latency.",
    color: "#b91c1c",
    bg: "#fef2f2",
  },
  {
    name: "HOT Tier",
    latency: "42ms",
    capacity: "5 Apps",
    role: "LRU Active Stack",
    description: "Maintains the most recently used application list. Operates via standard Least Recently Used (LRU) eviction to capture direct temporal back-and-forth loops.",
    color: "#ea580c",
    bg: "#fff7ed",
  },
  {
    name: "WARM Tier",
    latency: "190ms",
    capacity: "8 Apps",
    role: "AI Prefetched Standby",
    description: "Houses apps pre-loaded by the GraphMind V6 recommendation pipeline. Recommended candidate apps are brought from disk to RAM before the user requests them.",
    color: "#2563eb",
    bg: "#eff6ff",
  },
  {
    name: "COOL Tier",
    latency: "400ms",
    capacity: "20 Apps",
    role: "Compressed RAM Standby",
    description: "A signature V6 innovation. Instead of direct eviction to disk, apps evicted from HOT/WARM are held in a compressed RAM memory tier (zRAM), reducing cold launches by 44%.",
    color: "#0891b2",
    bg: "#ecfeff",
  },
  {
    name: "COLD Tier",
    latency: "720ms",
    capacity: "Infinite",
    role: "Flash Storage / Disk",
    description: "Apps saved on physical storage. Loading an app from COLD results in a full cold launch with high system overhead, battery consumption, and latency.",
    color: "#4b5563",
    bg: "#f9fafb",
  },
];

const COMPONENT_DETAILS = {
  graph: {
    title: "Markov Transition Graph Engine",
    desc: "Maintains a running probability matrix of app transitions tailored individually to each device user. Updates in real-time on the EventBus to capture high-order sequence chains.",
    icon: Database,
    stats: [
      { label: "Graph Node Limit", val: "1,000 per user" },
      { label: "Update Complexity", val: "O(1) amortized" },
    ],
  },
  transformer: {
    title: "Embedding Transformer Reranker",
    desc: "Uses small, isolated, per-user Transformer sequence models (~585KB each). Embeds 34-dimensional app taxonomy and state metrics to output highly personalized next-app launch probabilities.",
    icon: Sparkles,
    stats: [
      { label: "Model Weight Size", val: "~585 KB" },
      { label: "Features Evaluated", val: "Category, Frequency, Recency" },
    ],
  },
  rl: {
    title: "PPO Policy Controller",
    desc: "A reinforcement learning agent trained with Proximal Policy Optimization (PPO). Evaluates a rolling 20-step cache hit rate and dynamically adjusts the prefetch confidence thresholds (±0.005) to optimize RAM use against battery costs.",
    icon: Cpu,
    stats: [
      { label: "Reward Function", val: "Hits - Thrash - Battery" },
      { label: "Action Range", val: "Threshold scale [0.0, 1.0]" },
    ],
  },
  security: {
    title: "Context Boundary Enforcer",
    desc: "Subscribes to the EventBus and detects boundaries between sensitive (Financial, Health, Enterprise) and open (Social, Gaming) application domains. Immediately flushes cache regions during high-to-low transitions to prevent data snooping.",
    icon: Shield,
    stats: [
      { label: "Detection Latency", val: "< 1ms" },
      { label: "Target Categories", val: "Financial, Health, Enterprise" },
    ],
  },
};

export default function Architecture() {
  const [selectedComponent, setSelectedComponent] = useState<keyof typeof COMPONENT_DETAILS>("transformer");
  const [selectedTier, setSelectedTier] = useState<number>(3); // Default to COOL

  return (
    <div className="max-w-5xl mx-auto px-8 py-10">
      
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
        <div className="flex items-center gap-2 text-xs text-gray-400 mb-3">
          <span>Samsung EnnovateX AX 2026</span>
          <span>·</span>
          <span>Technical Architecture</span>
        </div>
        <h1 className="text-2xl font-semibold text-gray-900 mb-1.5">GraphMind V6 Working & Architecture</h1>
        <p className="text-sm text-gray-500 max-w-xl">
          Detailed overview of the intelligent 5-Tier Memory Cache hierarchy, per-user sequence Transformers, Reinforcement Learning adjustments, and security flushes.
        </p>
      </motion.div>

      {/* Pipeline Visual Diagram */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="card p-6 mb-8"
      >
        <h2 className="section-title mb-4">Pipeline Execution Flow</h2>
        <div className="flex flex-col items-center max-w-xl mx-auto">
          
          <div className="w-full flex items-center gap-4 p-3.5 rounded-lg border border-gray-200 bg-gray-50">
            <div className="p-2.5 rounded-lg bg-white border border-gray-200 flex-shrink-0 flex items-center justify-center"><Server size={18} className="text-gray-500" /></div>
            <div>
              <div className="text-xs font-semibold text-gray-900">1. Android EventBus</div>
              <p className="text-[10px] text-gray-500 mt-0.5">Intercepts app launches, battery updates, and weekend states.</p>
            </div>
          </div>

          <div className="h-4 w-px bg-gray-200" />

          <div className="w-full flex items-center gap-4 p-3.5 rounded-lg border border-gray-200 bg-gray-50">
            <div className="p-2.5 rounded-lg bg-white border border-gray-200 flex-shrink-0 flex items-center justify-center"><Database size={18} className="text-blue-500" /></div>
            <div>
              <div className="text-xs font-semibold text-gray-900">2. Markov Graph Engine</div>
              <p className="text-[10px] text-gray-500 mt-0.5">Updates individual user app transition weights and state counts.</p>
            </div>
          </div>

          <div className="h-4 w-px bg-gray-200" />

          <div className="w-full flex items-center gap-4 p-3.5 rounded-lg border border-blue-200 bg-blue-50">
            <div className="p-2.5 rounded-lg bg-white border border-blue-200 flex-shrink-0 flex items-center justify-center"><Sparkles size={18} className="text-blue-600" /></div>
            <div>
              <div className="text-xs font-semibold text-blue-900">3. V6 Embedding Transformer</div>
              <p className="text-[10px] text-blue-600/70 mt-0.5">Reranks transition candidates using per-user embedding sequence models.</p>
            </div>
          </div>

          <div className="h-4 w-px bg-gray-200" />

          <div className="w-full flex items-center gap-4 p-3.5 rounded-lg border border-gray-200 bg-gray-50">
            <div className="p-2.5 rounded-lg bg-white border border-gray-200 flex-shrink-0 flex items-center justify-center"><Cpu size={18} className="text-amber-500" /></div>
            <div>
              <div className="text-xs font-semibold text-gray-900">4. RL (PPO) Adjuster</div>
              <p className="text-[10px] text-gray-500 mt-0.5">Sets prefetch threshold dynamically on 20-step rolling success.</p>
            </div>
          </div>

          <div className="h-4 w-px bg-gray-200" />

          <div className="w-full flex items-center gap-4 p-3.5 rounded-lg border border-green-200 bg-green-50">
            <div className="p-2.5 rounded-lg bg-white border border-green-200 flex-shrink-0 flex items-center justify-center"><Layers size={18} className="text-green-600" /></div>
            <div>
              <div className="text-xs font-semibold text-green-900">5. 5-Tier Cache</div>
              <p className="text-[10px] text-green-600/70 mt-0.5">Loads predicted apps to WARM tier. Compresses evicted ones to COOL standby.</p>
            </div>
          </div>

        </div>
      </motion.div>

      {/* Grid of 5-Tier Cache Stack & Component Deep Dive */}
      <div className="grid grid-cols-2 gap-6 mb-8">
        
        {/* Interactive 5-Tier Cache Stack */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card p-5"
        >
          <h2 className="section-title mb-1">5-Tier Memory Cache Stack</h2>
          <p className="text-xs text-gray-400 mb-4">Click a tier to inspect its latency and capacity characteristics</p>
          
          <div className="flex flex-col gap-2 mb-4">
            {TIERS.map((tier, idx) => {
              const isSelected = selectedTier === idx;
              return (
                <div
                  key={tier.name}
                  onClick={() => setSelectedTier(idx)}
                  className="cursor-pointer p-3 rounded-lg border transition-all flex items-center justify-between"
                  style={{
                    backgroundColor: isSelected ? tier.bg : "#ffffff",
                    borderColor: isSelected ? tier.color : "#e5e7eb",
                    borderWidth: isSelected ? "2px" : "1px",
                  }}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: tier.color }}
                    />
                    <div>
                      <div className="text-sm font-semibold text-gray-900">{tier.name}</div>
                      <div className="text-[10px] text-gray-400">{tier.role}</div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-mono font-semibold" style={{ color: tier.color }}>{tier.latency}</div>
                    <div className="text-[10px] text-gray-400">{tier.capacity}</div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Tier Detail Card */}
          <div className="p-3.5 rounded-lg border border-gray-100 bg-gray-50">
            <div className="flex items-center gap-2 mb-1.5">
              <Info size={14} className="text-blue-500" />
              <div className="text-xs font-semibold text-gray-900">
                Detailed Working: {TIERS[selectedTier].name}
              </div>
            </div>
            <p className="text-xs text-gray-600 leading-relaxed">
              {TIERS[selectedTier].description}
            </p>
          </div>
        </motion.div>

        {/* Component Deep Dives */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.12 }}
          className="card p-5 flex flex-col justify-between"
        >
          <div>
            <h2 className="section-title mb-1">GraphMind Core AI Components</h2>
            <p className="text-xs text-gray-400 mb-4">Click to inspect individual models and decision systems</p>

            <div className="grid grid-cols-2 gap-2 mb-4">
              {(Object.keys(COMPONENT_DETAILS) as Array<keyof typeof COMPONENT_DETAILS>).map((key) => {
                const cmp = COMPONENT_DETAILS[key];
                const isSelected = selectedComponent === key;
                return (
                  <button
                    key={key}
                    onClick={() => setSelectedComponent(key)}
                    className={`flex items-center gap-2.5 p-3 rounded-lg border text-left transition-colors ${
                      isSelected
                        ? "bg-gray-900 border-gray-900 text-white"
                        : "bg-white border-gray-200 text-gray-700 hover:bg-gray-50"
                    }`}
                  >
                    <cmp.icon size={15} className={isSelected ? "text-white" : "text-gray-400"} />
                    <span className="text-xs font-medium">{cmp.title.split(" ")[0]} {cmp.title.split(" ")[1]}</span>
                  </button>
                );
              })}
            </div>

            <div className="p-4 rounded-lg border border-gray-100 bg-gray-50 flex-1">
              <div className="flex items-center gap-2 mb-2">
                {(() => {
                  const Icon = COMPONENT_DETAILS[selectedComponent].icon;
                  return <Icon size={16} className="text-blue-500" />;
                })()}
                <h3 className="text-sm font-semibold text-gray-900">
                  {COMPONENT_DETAILS[selectedComponent].title}
                </h3>
              </div>
              <p className="text-xs text-gray-600 leading-relaxed mb-4">
                {COMPONENT_DETAILS[selectedComponent].desc}
              </p>

              <div className="grid grid-cols-2 gap-3 pt-3 border-t border-gray-200">
                {COMPONENT_DETAILS[selectedComponent].stats.map((st) => (
                  <div key={st.label}>
                    <div className="text-[10px] text-gray-400 uppercase tracking-wider">{st.label}</div>
                    <div className="text-xs font-mono font-semibold text-gray-800 mt-0.5">{st.val}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-4 p-3 rounded-lg border border-yellow-100 bg-yellow-50 flex items-center gap-2">
            <Sparkles size={14} className="text-yellow-600 flex-shrink-0" />
            <div className="text-[11px] text-yellow-800">
              <strong>Gemma Explainability:</strong> Predictions are explained dynamically (e.g. why we pre-fetch Slack) by mapping graph weights & RL state representations to human-readable text.
            </div>
          </div>
        </motion.div>

      </div>

      {/* Security Context Isolation Segment */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        className="card p-5"
      >
        <div className="flex items-start gap-4">
          <div className="p-2 bg-red-50 text-red-600 rounded-lg">
            <Shield size={20} />
          </div>
          <div className="flex-1">
            <h2 className="section-title mb-1.5">Context Boundary Security Isolation</h2>
            <p className="text-xs text-gray-500 mb-3">
              GraphMind V6 features dedicated memory isolation that prevents cache side-channel attacks by clearing recommended states during security transitions.
            </p>
            <div className="grid grid-cols-3 gap-3">
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100">
                <div className="text-xs font-semibold text-gray-900 mb-1 flex items-center gap-1.5">
                  <Key size={12} className="text-red-500" />
                  Isolation Boundaries
                </div>
                <p className="text-[11px] text-gray-600">
                  Separates sensitive user domains (Finance, Enterprise, Health) from standard ones (Social, Entertainment).
                </p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100">
                <div className="text-xs font-semibold text-gray-900 mb-1 flex items-center gap-1.5">
                  <Zap size={12} className="text-amber-500" />
                  Instant Cache Flush
                </div>
                <p className="text-[11px] text-gray-600">
                  Clears the WARM (prefetch) cache and evicts background RAM allocations on transition to prevent app profiling.
                </p>
              </div>
              <div className="p-3 bg-gray-50 rounded-lg border border-gray-100">
                <div className="text-xs font-semibold text-gray-900 mb-1 flex items-center gap-1.5">
                  <CheckCircle2 size={12} className="text-green-500" />
                  Security Verification
                </div>
                <p className="text-[11px] text-gray-600">
                  Subscribed directly to EventBus launches to ensure flushes complete within milliseconds of domain switching.
                </p>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

    </div>
  );
}
