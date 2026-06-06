"use client";
import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Pause, SkipForward, RotateCcw } from "lucide-react";

interface Pred { app: string; short: string; confidence: number; }
interface Event {
  step: number; timestamp: string; app: string; short: string;
  tier: "hot" | "warm" | "miss"; hit: boolean; latency_saved: number;
  threshold: number; predictions: Pred[];
  hot_cache: string[]; warm_cache: string[];
}
interface UserData { user_id: string; n_events: number; events: Event[]; }

export default function Simulator() {
  const [all, setAll] = useState<Record<string, UserData>>({});
  const [uid, setUid] = useState("");
  const [events, setEvents] = useState<Event[]>([]);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(500);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const feedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/data/transitions.json").then(r => r.json()).then((d: Record<string, UserData>) => {
      setAll(d);
      const k = Object.keys(d)[0];
      if (k) { setUid(k); setEvents(d[k].events); }
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!uid || !all[uid]) return;
    setEvents(all[uid].events); setStep(0); setPlaying(false);
  }, [uid]);

  useEffect(() => {
    if (playing) {
      timer.current = setInterval(() => {
        setStep(s => { if (s >= events.length - 1) { setPlaying(false); return s; } return s + 1; });
      }, speed);
    }
    return () => { if (timer.current) clearInterval(timer.current); };
  }, [playing, speed, events.length]);

  useEffect(() => {
    if (feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
  }, [step]);

  const cur = events[step];
  const history = events.slice(Math.max(0, step - 15), step + 1);
  const hits = events.slice(0, step + 1).filter(e => e.hit).length;
  const saved = events.slice(0, step + 1).reduce((s, e) => s + e.latency_saved, 0);
  const hr = step >= 0 ? hits / (step + 1) : 0;

  return (
    <div className="max-w-5xl mx-auto px-8 py-10">
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mb-5">
        <h1 className="page-title mb-1">Cache Simulator</h1>
        <p className="text-sm text-gray-500">
          GraphMindRL_V5 · HOT = 5 · WARM = 15 · Adaptive threshold · Real UbiqLog sequences
        </p>
      </motion.div>

      {/* Controls */}
      <div className="card p-3.5 mb-5 flex flex-wrap gap-3 items-center">
        <select value={uid} onChange={e => setUid(e.target.value)} className="input text-xs">
          {Object.keys(all).map(k => <option key={k} value={k}>{k}</option>)}
        </select>
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <span>Speed</span>
          <input type="range" min={100} max={2000} step={100} value={speed}
            onChange={e => setSpeed(Number(e.target.value))} className="w-20 accent-gray-700" />
          <span className="mono text-gray-700 w-14">{speed}ms</span>
        </div>
        <div className="flex gap-1.5">
          <button onClick={() => { setStep(0); setPlaying(false); }} className="btn btn-secondary p-2"><RotateCcw size={14} /></button>
          <button onClick={() => setPlaying(p => !p)} className="btn btn-primary">
            {playing ? <><Pause size={14} /> Pause</> : <><Play size={14} /> Play</>}
          </button>
          <button onClick={() => setStep(s => Math.min(events.length - 1, s + 1))} className="btn btn-secondary p-2"><SkipForward size={14} /></button>
        </div>
        <div className="flex-1 flex items-center gap-2 ml-2">
          <div className="flex-1 h-1.5 rounded-full bg-gray-100">
            <div className="h-1.5 rounded-full bg-gray-800 transition-all"
                 style={{ width: `${(step / Math.max(events.length - 1, 1)) * 100}%` }} />
          </div>
          <span className="text-xs mono text-gray-400">{step + 1}/{events.length}</span>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-3 mb-5">
        {[
          { label: "Cache Hits", value: `${hits}`, sub: `of ${step + 1}` },
          { label: "Hit Rate", value: `${(hr * 100).toFixed(1)}%`, sub: "rolling", green: hr > 0.8 },
          { label: "Latency Saved", value: `${(saved / 1000).toFixed(1)}s`, sub: `${saved.toLocaleString()}ms total` },
          { label: "Threshold", value: cur ? cur.threshold.toFixed(3) : "—", sub: "adaptive ±0.005" },
        ].map(s => (
          <div key={s.label} className="card p-3.5">
            <div className="label mb-1.5">{s.label}</div>
            <div className={`text-lg font-semibold mono ${s.green ? "text-green-600" : "text-gray-900"}`}>{s.value}</div>
            <div className="text-xs text-gray-400 mt-0.5">{s.sub}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-3 gap-4">
        {/* Cache state */}
        <div className="space-y-3">
          <div className="card p-4">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-2 h-2 rounded-full bg-orange-400" />
              <span className="text-sm font-medium text-gray-900">HOT Cache</span>
              <span className="text-xs text-gray-400 ml-auto">5 slots · RAM</span>
            </div>
            <div className="space-y-1.5">
              {Array.from({ length: 5 }).map((_, i) => {
                const app = cur?.hot_cache[i];
                return (
                  <div key={i} className="flex items-center gap-2">
                    <span className="text-xs text-gray-300 w-3">{i + 1}</span>
                    {app
                      ? <AnimatePresence mode="wait"><motion.div key={app} initial={{ opacity: 0, x: -4 }} animate={{ opacity: 1, x: 0 }}
                          className="flex-1 px-2.5 py-1.5 rounded-md text-xs font-medium bg-orange-50 text-orange-800 border border-orange-100">
                          {app}
                        </motion.div></AnimatePresence>
                      : <div className="flex-1 px-2.5 py-1.5 rounded-md text-xs text-gray-300 border border-dashed border-gray-200">empty</div>
                    }
                  </div>
                );
              })}
            </div>
          </div>

          <div className="card p-4">
            <div className="flex items-center gap-2 mb-3">
              <div className="w-2 h-2 rounded-full bg-blue-400" />
              <span className="text-sm font-medium text-gray-900">WARM Cache</span>
              <span className="text-xs text-gray-400 ml-auto">15 slots</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              <AnimatePresence>
                {(cur?.warm_cache || []).map(app => (
                  <motion.span key={app} initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    className="px-2 py-1 rounded text-xs bg-blue-50 text-blue-700 border border-blue-100">
                    {app}
                  </motion.span>
                ))}
                {(!cur?.warm_cache?.length) && <span className="text-xs text-gray-400">empty</span>}
              </AnimatePresence>
            </div>
          </div>
        </div>

        {/* Current event */}
        <div className="space-y-3">
          <AnimatePresence mode="wait">
            {cur && (
              <motion.div key={step} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }} transition={{ duration: 0.15 }}
                className={`card p-4 ${cur.hit ? "border-green-200" : "border-red-100"}`}>
                <div className="text-xs text-gray-400 mb-2">{cur.timestamp} · step {step + 1}</div>
                <div className="text-lg font-semibold text-gray-900 mb-0.5">{cur.short}</div>
                <div className="mono text-xs text-gray-400 mb-3 break-all">{cur.app}</div>
                <div className="flex gap-2 mb-3">
                  <span className={cur.tier === "hot" ? "pill-hot" : cur.tier === "warm" ? "pill-warm" : "pill-miss"}>
                    {cur.tier.toUpperCase()}
                  </span>
                  <span className={`badge ${cur.hit ? "badge-green" : "badge-red"}`}>
                    {cur.hit ? "HIT" : "MISS"}
                  </span>
                  {cur.latency_saved > 0 && (
                    <span className="badge badge-amber">{cur.latency_saved.toLocaleString()}ms saved</span>
                  )}
                </div>
                <div>
                  <div className="label mb-2">Predictions (thresh = {cur.threshold.toFixed(3)})</div>
                  {cur.predictions.length === 0
                    ? <div className="text-xs text-gray-400">No candidates above threshold</div>
                    : cur.predictions.map((p, i) => (
                      <div key={p.app} className="flex items-center gap-2 mb-1.5">
                        <span className="text-xs text-gray-400 w-3">{i + 1}</span>
                        <span className="text-xs text-gray-700 w-24 truncate">{p.short}</span>
                        <div className="flex-1 progress-bar">
                          <div className="progress-fill bg-gray-800" style={{ width: `${p.confidence * 100}%` }} />
                        </div>
                        <span className="mono text-xs text-gray-600 w-10 text-right">{(p.confidence * 100).toFixed(0)}%</span>
                      </div>
                    ))
                  }
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Feed */}
        <div className="card overflow-hidden flex flex-col">
          <div className="px-4 py-3 text-sm font-medium text-gray-900" style={{ borderBottom: "1px solid #f3f4f6" }}>
            Event Feed
          </div>
          <div ref={feedRef} className="flex-1 overflow-y-auto" style={{ maxHeight: 440 }}>
            {history.map((e, i) => (
              <div key={e.step}
                className={`px-4 py-2 text-xs flex items-center gap-2.5 ${i === history.length - 1 ? "bg-gray-50" : ""}`}
                style={{ borderBottom: "1px solid #f9fafb" }}>
                <span className="text-gray-300 mono w-5">{e.step + 1}</span>
                <span className="text-gray-700 flex-1 truncate">{e.short}</span>
                <span className={e.tier === "hot" ? "pill-hot" : e.tier === "warm" ? "pill-warm" : "pill-miss"}>
                  {e.tier}
                </span>
                {e.latency_saved > 0 && (
                  <span className="mono text-gray-500">{e.latency_saved.toLocaleString()}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
