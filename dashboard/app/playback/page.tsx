"use client";
import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Play, Pause, SkipForward, RotateCcw } from "lucide-react";

interface Pred { app: string; short: string; confidence: number; }
interface Event {
  step: number; timestamp: string; app: string; short: string;
  tier: "hot" | "warm" | "miss"; hit: boolean; latency_saved: number;
  threshold: number; predictions: Pred[];
  hot_cache: string[]; warm_cache: string[];
}
interface UserData { user_id: string; n_events: number; events: Event[]; }

export default function Playback() {
  const [all, setAll] = useState<Record<string, UserData>>({});
  const [uid, setUid] = useState("");
  const [events, setEvents] = useState<Event[]>([]);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(400);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    fetch("/data/transitions.json").then(r => r.json()).then((d: Record<string, UserData>) => {
      setAll(d); const k = Object.keys(d)[0];
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

  const cur = events[step];
  const window30 = events.slice(Math.max(0, step - 30), step + 1).map((e, i) => ({
    i: Math.max(0, step - 30) + i,
    hr: events.slice(0, Math.max(0, step - 30) + i + 1).filter(x => x.hit).length / (Math.max(0, step - 30) + i + 1),
    threshold: e.threshold,
  }));
  const totalHits = events.slice(0, step + 1).filter(e => e.hit).length;
  const totalSaved = events.slice(0, step + 1).reduce((s, e) => s + e.latency_saved, 0);

  return (
    <div className="max-w-5xl mx-auto px-8 py-10">
      <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="mb-5">
        <h1 className="page-title mb-1">User Playback</h1>
        <p className="text-sm text-gray-500">Step through real user event sequences · Watch the RL controller adapt in real time</p>
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
            {playing ? <><Pause size={14} />Pause</> : <><Play size={14} />Play</>}
          </button>
          <button onClick={() => setStep(s => Math.min(events.length - 1, s + 1))} className="btn btn-secondary p-2"><SkipForward size={14} /></button>
        </div>
        <div className="flex-1 flex items-center gap-3 ml-2">
          <input type="range" min={0} max={Math.max(events.length - 1, 1)} value={step}
            onChange={e => { setStep(Number(e.target.value)); setPlaying(false); }}
            className="flex-1 accent-gray-700" />
          <span className="mono text-xs text-gray-400 whitespace-nowrap">{step + 1}/{events.length}</span>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-4 gap-3 mb-5">
        {[
          { label: "Step", value: `${step + 1}`, sub: `of ${events.length}` },
          { label: "Hit Rate", value: `${(totalHits / Math.max(step + 1, 1) * 100).toFixed(1)}%`, sub: "cumulative", green: true },
          { label: "Latency Saved", value: `${(totalSaved / 1000).toFixed(1)}s`, sub: `${totalSaved.toLocaleString()}ms` },
          { label: "Threshold", value: cur ? cur.threshold.toFixed(3) : "—", sub: "current value" },
        ].map(s => (
          <div key={s.label} className="card p-3.5">
            <div className="label mb-1.5">{s.label}</div>
            <div className={`text-lg font-semibold mono ${s.green ? "text-green-600" : "text-gray-900"}`}>{s.value}</div>
            <div className="text-xs text-gray-400 mt-0.5">{s.sub}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-5 gap-4">
        {/* Event detail */}
        <div className="col-span-2 space-y-3">
          <AnimatePresence mode="wait">
            {cur && (
              <motion.div key={step} initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }} transition={{ duration: 0.12 }}
                className={`card p-4 ${cur.hit ? "border-green-200" : ""}`}>
                <div className="text-xs text-gray-400 mb-2">{cur.timestamp}</div>
                <div className="text-xl font-semibold text-gray-900">{cur.short}</div>
                <div className="mono text-xs text-gray-400 mt-0.5 mb-3 break-all">{cur.app}</div>
                <div className="flex gap-2 flex-wrap mb-4">
                  <span className={cur.tier === "hot" ? "pill-hot" : cur.tier === "warm" ? "pill-warm" : "pill-miss"}>{cur.tier.toUpperCase()}</span>
                  <span className={`badge ${cur.hit ? "badge-green" : "badge-red"}`}>{cur.hit ? "HIT" : "MISS"}</span>
                  {cur.latency_saved > 0 && <span className="badge badge-amber">{cur.latency_saved.toLocaleString()}ms saved</span>}
                </div>

                <div className="label mb-2">Top predictions</div>
                {cur.predictions.length === 0
                  ? <div className="text-xs text-gray-400">No candidates above {cur.threshold.toFixed(3)}</div>
                  : cur.predictions.map((p, i) => (
                    <div key={p.app} className="flex items-center gap-2 mb-1.5">
                      <span className="text-xs text-gray-400 w-3">{i + 1}</span>
                      <span className="text-xs text-gray-700 w-20 truncate">{p.short}</span>
                      <div className="flex-1 progress-bar">
                        <div className="progress-fill bg-gray-800" style={{ width: `${p.confidence * 100}%` }} />
                      </div>
                      <span className="mono text-xs text-gray-600 w-8 text-right">{(p.confidence * 100).toFixed(0)}%</span>
                    </div>
                  ))
                }
              </motion.div>
            )}
          </AnimatePresence>

          {cur && (
            <div className="card p-4">
              <div className="label mb-2.5">Cache snapshot</div>
              <div className="flex items-center gap-2 flex-wrap mb-1.5">
                <span className="text-xs font-medium text-orange-600 w-10">HOT</span>
                {cur.hot_cache.map(a => <span key={a} className="pill-hot">{a}</span>)}
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-xs font-medium text-blue-600 w-10">WARM</span>
                {cur.warm_cache.map(a => <span key={a} className="pill-warm">{a}</span>)}
              </div>
            </div>
          )}
        </div>

        {/* Charts */}
        <div className="col-span-3 space-y-4">
          <div className="card p-4">
            <div className="section-title mb-4">Rolling Hit Rate</div>
            <ResponsiveContainer width="100%" height={150}>
              <AreaChart data={window30}>
                <defs>
                  <linearGradient id="hrG" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#111827" stopOpacity={0.08} />
                    <stop offset="95%" stopColor="#111827" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid vertical={false} stroke="#f3f4f6" />
                <XAxis dataKey="i" tick={{ fill: "#9ca3af", fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} tick={{ fill: "#9ca3af", fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, "Hit Rate"]}
                         contentStyle={{ background: "white", border: "1px solid #e5e7eb", borderRadius: 6, fontSize: 12 }} />
                <Area type="monotone" dataKey="hr" stroke="#111827" fill="url(#hrG)" strokeWidth={1.5} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="card p-4">
            <div className="section-title mb-1">Adaptive Threshold (RL Controller)</div>
            <p className="text-xs text-gray-400 mb-4">Rises when hit rate {'>'} 80%, falls when {'<'} 50% · Step ±0.005</p>
            <ResponsiveContainer width="100%" height={150}>
              <AreaChart data={window30}>
                <defs>
                  <linearGradient id="thG" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.08} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid vertical={false} stroke="#f3f4f6" />
                <XAxis dataKey="i" tick={{ fill: "#9ca3af", fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis domain={[0.04, 0.26]} tick={{ fill: "#9ca3af", fontSize: 10 }} axisLine={false} tickLine={false} />
                <Tooltip formatter={(v: number) => [v.toFixed(3), "Threshold"]}
                         contentStyle={{ background: "white", border: "1px solid #e5e7eb", borderRadius: 6, fontSize: 12 }} />
                <Area type="monotone" dataKey="threshold" stroke="#3b82f6" fill="url(#thG)" strokeWidth={1.5} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
