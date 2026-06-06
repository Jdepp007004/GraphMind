"use client";
import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, SkipForward, RotateCcw, Zap, TrendingUp } from "lucide-react";

interface Prediction { app: string; short: string; confidence: number; trans_prob: number; }
interface Event {
  step: number; timestamp: string; app: string; short: string;
  tier: "hot" | "warm" | "miss"; hit: boolean; latency_saved: number;
  threshold: number; predictions: Prediction[];
  hot_cache: string[]; warm_cache: string[];
}
interface UserData { user_id: string; n_events: number; events: Event[]; }

const TIER_STYLE: Record<string, string> = {
  hot:  "bg-orange-500/20 border border-orange-500/40 text-orange-300",
  warm: "bg-blue-500/15 border border-blue-500/30 text-blue-300",
  miss: "bg-gray-600/15 border border-gray-600/20 text-gray-400",
};

const TIER_LABEL: Record<string, string> = {
  hot: "HOT ⚡", warm: "WARM 🌡", miss: "MISS ❌",
};

function CacheSlot({ label, tier }: { label: string; tier: "hot" | "warm" }) {
  return (
    <motion.div layout initial={{ scale: 0, opacity: 0 }} animate={{ scale: 1, opacity: 1 }}
      exit={{ scale: 0, opacity: 0 }} transition={{ type: "spring", stiffness: 400, damping: 25 }}
      className={`px-2 py-1 rounded text-xs font-medium text-center truncate ${tier === "hot" ? "tier-hot border" : "tier-warm border"}`}
      style={{ minWidth: "60px", maxWidth: "90px" }}>
      {label}
    </motion.div>
  );
}

export default function CacheSimulator() {
  const [allUsers, setAllUsers] = useState<Record<string, UserData>>({});
  const [userId, setUserId] = useState<string>("");
  const [events, setEvents] = useState<Event[]>([]);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(600);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const feedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/data/transitions.json").then(r => r.json()).then((d: Record<string, UserData>) => {
      setAllUsers(d);
      const firstKey = Object.keys(d)[0];
      if (firstKey) {
        setUserId(firstKey);
        setEvents(d[firstKey].events);
      }
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!userId || !allUsers[userId]) return;
    setEvents(allUsers[userId].events);
    setStep(0);
    setPlaying(false);
  }, [userId]);

  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(() => {
        setStep(s => {
          if (s >= events.length - 1) { setPlaying(false); return s; }
          return s + 1;
        });
      }, speed);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [playing, speed, events.length]);

  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = feedRef.current.scrollHeight;
    }
  }, [step]);

  const cur = events[step];
  const history = events.slice(Math.max(0, step - 12), step + 1);
  const totalHits = events.slice(0, step + 1).filter(e => e.hit).length;
  const totalSaved = events.slice(0, step + 1).reduce((s, e) => s + e.latency_saved, 0);

  return (
    <div className="min-h-screen p-8 grid-bg">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
        <h1 className="text-4xl font-black mb-2">Cache <span className="gradient-text">Simulator</span></h1>
        <p className="text-[#7a8fbf]">
          GraphMindRL_V5 live simulation · HOT=5 · WARM=15 · Adaptive threshold · Real UbiqLog sequences
        </p>
      </motion.div>

      {/* Controls */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
        className="glass-card p-4 mb-6 flex flex-wrap gap-4 items-center">
        <div className="flex items-center gap-2">
          <span className="text-xs text-[#7a8fbf]">User:</span>
          <select value={userId} onChange={e => setUserId(e.target.value)}
            className="bg-[#0a1430] border border-white/10 text-sm text-white rounded-lg px-3 py-1.5 outline-none">
            {Object.keys(allUsers).map(uid => (
              <option key={uid} value={uid}>{uid}</option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-[#7a8fbf]">Speed:</span>
          <input type="range" min={100} max={2000} step={100} value={speed}
            onChange={e => setSpeed(Number(e.target.value))} className="w-24 accent-blue-400" />
          <span className="text-xs font-mono text-[#00b0ff]">{speed}ms</span>
        </div>
        <div className="flex gap-2">
          <button onClick={() => { setStep(0); setPlaying(false); }}
            className="p-2 rounded-xl bg-white/5 hover:bg-white/10 text-[#7a8fbf] hover:text-white">
            <RotateCcw size={16} />
          </button>
          <button onClick={() => setPlaying(p => !p)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium
              ${playing ? "badge-production" : "badge-accepted"}`}>
            {playing ? <Pause size={14} /> : <Play size={14} />}
            {playing ? "Pause" : "Play"}
          </button>
          <button onClick={() => setStep(s => Math.min(events.length - 1, s + 1))}
            className="p-2 rounded-xl bg-white/5 hover:bg-white/10 text-[#7a8fbf] hover:text-white">
            <SkipForward size={16} />
          </button>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <div className="h-1.5 rounded-full bg-white/10 w-48">
            <div className="h-1.5 rounded-full transition-all" style={{ width: `${(step / Math.max(events.length - 1, 1)) * 100}%`, background: "linear-gradient(90deg, #1428a0, #00b0ff)" }} />
          </div>
          <span className="text-xs font-mono text-[#7a8fbf]">{step + 1}/{events.length}</span>
        </div>
      </motion.div>

      <div className="grid grid-cols-3 gap-4 mb-4">
        {[
          { label: "Cache Hits", value: totalHits, suffix: "", sub: `of ${step + 1} events`, color: "#00e676" },
          { label: "Hit Rate", value: step >= 0 ? ((totalHits / (step + 1)) * 100).toFixed(1) : "—", suffix: "%", sub: "rolling", color: "#00b0ff" },
          { label: "Latency Saved", value: Math.round(totalSaved / 1000), suffix: "s", sub: `${totalSaved.toLocaleString()}ms total`, color: "#ffa726" },
        ].map(s => (
          <div key={s.label} className="glass-card p-4 text-center">
            <div className="text-xs text-[#7a8fbf] mb-1">{s.label}</div>
            <div className="text-3xl font-black metric-number" style={{ color: s.color }}>
              {s.value}{s.suffix}
            </div>
            <div className="text-xs text-[#7a8fbf]">{s.sub}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-4">
        {/* Cache state */}
        <div className="space-y-4">
          {/* HOT */}
          <div className="glass-card p-4">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-2 h-2 rounded-full bg-orange-400 pulse" />
              <span className="font-bold text-sm text-orange-300">HOT Cache (5 slots)</span>
              <span className="ml-auto text-xs text-[#7a8fbf]">In RAM</span>
            </div>
            <div className="grid grid-cols-2 gap-2 min-h-[100px]">
              <AnimatePresence>
                {(cur?.hot_cache || []).map((app, i) => (
                  <CacheSlot key={app} label={app} tier="hot" />
                ))}
                {Array.from({ length: Math.max(0, 5 - (cur?.hot_cache?.length || 0)) }).map((_, i) => (
                  <div key={`empty-hot-${i}`}
                    className="px-2 py-1 rounded text-xs text-center border border-dashed border-white/10 text-white/20">
                    empty
                  </div>
                ))}
              </AnimatePresence>
            </div>
          </div>
          {/* WARM */}
          <div className="glass-card p-4">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-2 h-2 rounded-full bg-blue-400" />
              <span className="font-bold text-sm text-blue-300">WARM Cache (15 slots)</span>
              <span className="ml-auto text-xs text-[#7a8fbf]">Pre-loaded</span>
            </div>
            <div className="flex flex-wrap gap-1.5 min-h-[60px]">
              <AnimatePresence>
                {(cur?.warm_cache || []).map(app => (
                  <CacheSlot key={app} label={app} tier="warm" />
                ))}
              </AnimatePresence>
            </div>
          </div>
          {/* Threshold */}
          {cur && (
            <div className="glass-card p-4">
              <div className="text-xs text-[#7a8fbf] mb-2">Adaptive Threshold</div>
              <div className="font-mono text-2xl font-bold text-[#e040fb]">{cur.threshold.toFixed(3)}</div>
              <div className="h-2 rounded-full bg-white/10 mt-2">
                <div className="h-2 rounded-full" style={{ width: `${(cur.threshold / 0.25) * 100}%`, background: "linear-gradient(90deg, #e040fb, #7c4dff)" }} />
              </div>
              <div className="flex justify-between text-xs text-[#7a8fbf] mt-1">
                <span>0.05</span><span>0.25</span>
              </div>
            </div>
          )}
        </div>

        {/* Current event + predictions */}
        <div className="col-span-2 space-y-4">
          {cur && (
            <motion.div key={step} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
              className="glass-card p-5" style={{ borderColor: cur.hit ? "rgba(0,230,118,0.3)" : "rgba(255,23,68,0.2)" }}>
              <div className="flex items-start justify-between mb-4">
                <div>
                  <div className="text-xs text-[#7a8fbf] mb-1">Step {step + 1} · {cur.timestamp}</div>
                  <div className="text-2xl font-black text-white">{cur.short}</div>
                  <div className="text-xs font-mono text-[#7a8fbf] mt-0.5">{cur.app}</div>
                </div>
                <div className="text-right">
                  <div className={`text-2xl font-black ${cur.hit ? "text-[#00e676]" : "text-[#ff5252]"}`}>
                    {TIER_LABEL[cur.tier]}
                  </div>
                  {cur.latency_saved > 0 && (
                    <div className="flex items-center gap-1 text-[#ffa726] text-sm justify-end mt-1">
                      <Zap size={12} />
                      <span className="font-mono">{cur.latency_saved.toLocaleString()}ms saved</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Predictions */}
              <div>
                <div className="text-xs text-[#7a8fbf] mb-2">Top Predictions (confidence ≥ {cur.threshold.toFixed(3)})</div>
                <div className="space-y-1.5">
                  {cur.predictions.length === 0 && (
                    <div className="text-xs text-[#7a8fbf] italic">No candidates above threshold</div>
                  )}
                  {cur.predictions.map((p, i) => (
                    <div key={p.app} className="flex items-center gap-3">
                      <span className="text-xs text-[#7a8fbf] w-4">{i + 1}</span>
                      <span className="text-sm text-white w-32 truncate">{p.short}</span>
                      <div className="flex-1 h-1.5 rounded-full bg-white/10">
                        <motion.div initial={{ width: 0 }} animate={{ width: `${p.confidence * 100}%` }}
                          className="h-1.5 rounded-full"
                          style={{ background: `hsl(${120 + i * 30}, 70%, 50%)` }} />
                      </div>
                      <span className="font-mono text-xs w-12 text-right"
                            style={{ color: `hsl(${120 + i * 30}, 70%, 60%)` }}>
                        {(p.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {/* Event feed */}
          <div className="glass-card overflow-hidden">
            <div className="px-4 py-3 border-b border-white/5 flex items-center gap-2">
              <TrendingUp size={14} className="text-[#00b0ff]" />
              <span className="text-sm font-semibold">Event Feed</span>
            </div>
            <div ref={feedRef} className="overflow-y-auto" style={{ maxHeight: "260px" }}>
              {history.map((e, i) => (
                <div key={e.step}
                  className={`px-4 py-2 border-b border-white/5 flex items-center gap-3 text-xs transition-all
                    ${i === history.length - 1 ? "bg-white/5" : ""}`}>
                  <span className="text-[#7a8fbf] w-6 font-mono">{e.step + 1}</span>
                  <span className="text-white w-28 truncate">{e.short}</span>
                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${TIER_STYLE[e.tier]}`}>
                    {e.tier.toUpperCase()}
                  </span>
                  {e.latency_saved > 0 && (
                    <span className="text-[#ffa726] font-mono ml-auto">{e.latency_saved.toLocaleString()}ms</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
