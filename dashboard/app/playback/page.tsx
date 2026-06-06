"use client";
import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { Play, Pause, SkipForward, RotateCcw } from "lucide-react";

interface Prediction { app: string; short: string; confidence: number; trans_prob: number; }
interface Event {
  step: number; timestamp: string; app: string; short: string;
  tier: "hot" | "warm" | "miss"; hit: boolean; latency_saved: number;
  threshold: number; predictions: Prediction[];
  hot_cache: string[]; warm_cache: string[];
}
interface UserData { user_id: string; n_events: number; events: Event[]; }

export default function UserPlayback() {
  const [allUsers, setAllUsers] = useState<Record<string, UserData>>({});
  const [userId, setUserId] = useState<string>("");
  const [events, setEvents] = useState<Event[]>([]);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(400);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    fetch("/data/transitions.json").then(r => r.json()).then((d: Record<string, UserData>) => {
      setAllUsers(d);
      const firstKey = Object.keys(d)[0];
      if (firstKey) { setUserId(firstKey); setEvents(d[firstKey].events); }
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!userId || !allUsers[userId]) return;
    setEvents(allUsers[userId].events);
    setStep(0); setPlaying(false);
  }, [userId]);

  useEffect(() => {
    if (playing) {
      timerRef.current = setInterval(() => {
        setStep(s => { if (s >= events.length - 1) { setPlaying(false); return s; } return s + 1; });
      }, speed);
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [playing, speed, events.length]);

  const cur = events[step];
  const chartWindow = events.slice(Math.max(0, step - 30), step + 1).map((e, i) => ({
    i: Math.max(0, step - 30) + i,
    hr: events.slice(0, Math.max(0, step - 30) + i + 1).filter(x => x.hit).length / (Math.max(0, step - 30) + i + 1),
    threshold: e.threshold,
    saved: e.latency_saved / 1000,
  }));

  const overallHR = events.slice(0, step + 1).filter(e => e.hit).length / (step + 1);
  const totalSavedSec = events.slice(0, step + 1).reduce((s, e) => s + e.latency_saved, 0) / 1000;

  return (
    <div className="min-h-screen p-8 grid-bg">
      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
        <h1 className="text-4xl font-black mb-2">User <span className="gradient-text">Playback</span></h1>
        <p className="text-[#7a8fbf]">
          Step through real user app sequences · Watch GraphMindRL_V5 make predictions in real-time
        </p>
      </motion.div>

      {/* Controls */}
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}
        className="glass-card p-4 mb-6 flex flex-wrap gap-4 items-center">
        <select value={userId} onChange={e => setUserId(e.target.value)}
          className="bg-[#0a1430] border border-white/10 text-sm text-white rounded-lg px-3 py-1.5 outline-none">
          {Object.keys(allUsers).map(uid => (
            <option key={uid} value={uid}>User: {uid}</option>
          ))}
        </select>
        <div className="flex items-center gap-2">
          <span className="text-xs text-[#7a8fbf]">Speed:</span>
          <input type="range" min={100} max={2000} step={100} value={speed}
            onChange={e => setSpeed(Number(e.target.value))} className="w-24 accent-blue-400" />
          <span className="text-xs font-mono text-[#00b0ff]">{speed}ms/step</span>
        </div>
        <div className="flex gap-2">
          <button onClick={() => { setStep(0); setPlaying(false); }}
            className="p-2 rounded-xl bg-white/5 hover:bg-white/10 text-[#7a8fbf] hover:text-white">
            <RotateCcw size={16} />
          </button>
          <button onClick={() => setPlaying(p => !p)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium ${playing ? "badge-production" : "badge-accepted"}`}>
            {playing ? <Pause size={14} /> : <Play size={14} />}
            {playing ? "Pause" : "Play"}
          </button>
          <button onClick={() => setStep(s => Math.min(events.length - 1, s + 1))}
            className="p-2 rounded-xl bg-white/5 hover:bg-white/10 text-[#7a8fbf] hover:text-white">
            <SkipForward size={16} />
          </button>
        </div>
        <div className="ml-auto flex-1 flex items-center gap-3">
          <input type="range" min={0} max={events.length - 1} value={step}
            onChange={e => { setStep(Number(e.target.value)); setPlaying(false); }}
            className="flex-1 accent-blue-400" />
          <span className="text-xs font-mono text-[#7a8fbf] whitespace-nowrap">
            {step + 1} / {events.length}
          </span>
        </div>
      </motion.div>

      {/* Stats row */}
      <div className="grid grid-cols-4 gap-3 mb-4">
        {[
          { label: "Current Step", value: `${step + 1}`, color: "#7a8fbf" },
          { label: "Hit Rate", value: `${(overallHR * 100).toFixed(1)}%`, color: "#00e676" },
          { label: "Latency Saved", value: `${totalSavedSec.toFixed(1)}s`, color: "#ffa726" },
          { label: "Threshold", value: cur ? cur.threshold.toFixed(3) : "—", color: "#e040fb" },
        ].map(s => (
          <div key={s.label} className="glass-card p-3 text-center">
            <div className="text-xs text-[#7a8fbf] mb-1">{s.label}</div>
            <div className="text-xl font-black font-mono" style={{ color: s.color }}>{s.value}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-5 gap-4">
        {/* Event Detail */}
        <div className="col-span-2 space-y-4">
          <AnimatePresence mode="wait">
            {cur && (
              <motion.div key={step} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }} transition={{ duration: 0.2 }}
                className="glass-card p-5" style={{ borderColor: cur.hit ? "rgba(0,230,118,0.35)" : "rgba(255,23,68,0.25)" }}>
                <div className="flex items-center gap-3 mb-4">
                  <div className={`w-3 h-3 rounded-full ${cur.hit ? "bg-[#00e676]" : "bg-[#ff5252]"} pulse`} />
                  <span className="text-xs text-[#7a8fbf]">{cur.timestamp}</span>
                </div>
                <div className="text-3xl font-black text-white mb-1">{cur.short}</div>
                <div className="text-xs font-mono text-[#7a8fbf] mb-4 break-all">{cur.app}</div>

                <div className="flex gap-3 mb-4">
                  <div className={`flex-1 py-3 px-4 rounded-xl text-center font-bold ${
                    cur.tier === "hot" ? "tier-hot border" : cur.tier === "warm" ? "tier-warm border" : "tier-cold border"}`}>
                    <div className="text-xs opacity-70 mb-0.5">Cache Tier</div>
                    <div>{cur.tier.toUpperCase()}</div>
                  </div>
                  <div className={`flex-1 py-3 px-4 rounded-xl text-center font-bold border
                    ${cur.hit ? "bg-green-500/15 border-green-500/30 text-green-300" : "bg-red-500/12 border-red-500/20 text-red-300"}`}>
                    <div className="text-xs opacity-70 mb-0.5">Result</div>
                    <div>{cur.hit ? "HIT ✓" : "MISS ✗"}</div>
                  </div>
                  {cur.latency_saved > 0 && (
                    <div className="flex-1 py-3 px-4 rounded-xl text-center font-bold bg-orange-500/15 border border-orange-500/30 text-orange-300">
                      <div className="text-xs opacity-70 mb-0.5">Saved</div>
                      <div>{cur.latency_saved.toLocaleString()}ms</div>
                    </div>
                  )}
                </div>

                <div>
                  <div className="text-xs text-[#7a8fbf] mb-2">Confidence Scores (threshold = {cur.threshold.toFixed(3)})</div>
                  {cur.predictions.length === 0 ? (
                    <div className="text-xs text-[#7a8fbf] italic">No candidates above threshold</div>
                  ) : cur.predictions.map((p, i) => (
                    <div key={p.app} className="flex items-center gap-2 mb-1.5">
                      <span className="text-xs text-[#7a8fbf] w-3">{i + 1}</span>
                      <span className="text-xs text-white w-20 truncate">{p.short}</span>
                      <div className="flex-1 h-1.5 rounded-full bg-white/10 overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }} animate={{ width: `${p.confidence * 100}%` }}
                          className="h-1.5 rounded-full"
                          style={{ background: `hsl(${140 + i * 25}, 70%, 50%)` }} />
                      </div>
                      <span className="text-xs font-mono w-10 text-right"
                            style={{ color: `hsl(${140 + i * 25}, 70%, 60%)` }}>
                        {(p.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Cache snapshot */}
          {cur && (
            <div className="glass-card p-4">
              <div className="text-xs text-[#7a8fbf] mb-2 font-medium">Cache State</div>
              <div className="flex items-center gap-1.5 flex-wrap mb-2">
                <span className="text-xs text-orange-400 font-medium w-8">HOT</span>
                {cur.hot_cache.map(a => (
                  <span key={a} className="px-2 py-0.5 rounded text-xs tier-hot border font-medium">{a}</span>
                ))}
              </div>
              <div className="flex items-center gap-1.5 flex-wrap">
                <span className="text-xs text-blue-400 font-medium w-8">WARM</span>
                {cur.warm_cache.map(a => (
                  <span key={a} className="px-2 py-0.5 rounded text-xs tier-warm border">{a}</span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Charts */}
        <div className="col-span-3 space-y-4">
          {/* Hit rate chart */}
          <div className="glass-card p-4">
            <h3 className="text-sm font-semibold mb-3 text-[#00b0ff]">Rolling Hit Rate</h3>
            <ResponsiveContainer width="100%" height={160}>
              <AreaChart data={chartWindow}>
                <defs>
                  <linearGradient id="hrGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#00e676" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#00e676" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="i" tick={{ fill: "#7a8fbf", fontSize: 10 }} />
                <YAxis domain={[0, 1]} tickFormatter={v => `${(v*100).toFixed(0)}%`} tick={{ fill: "#7a8fbf", fontSize: 10 }} />
                <Tooltip formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, "Hit Rate"]}
                         contentStyle={{ background: "rgba(5,9,23,0.95)", border: "1px solid rgba(0,176,255,0.3)", borderRadius: "10px" }} />
                <Area type="monotone" dataKey="hr" stroke="#00e676" fill="url(#hrGrad)" strokeWidth={2} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Threshold chart */}
          <div className="glass-card p-4">
            <h3 className="text-sm font-semibold mb-3 text-[#e040fb]">Adaptive Threshold (RL Controller)</h3>
            <ResponsiveContainer width="100%" height={160}>
              <AreaChart data={chartWindow}>
                <defs>
                  <linearGradient id="threshGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#e040fb" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#e040fb" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="i" tick={{ fill: "#7a8fbf", fontSize: 10 }} />
                <YAxis domain={[0.04, 0.26]} tick={{ fill: "#7a8fbf", fontSize: 10 }} />
                <Tooltip formatter={(v: number) => [v.toFixed(3), "Threshold"]}
                         contentStyle={{ background: "rgba(5,9,23,0.95)", border: "1px solid rgba(224,64,251,0.3)", borderRadius: "10px" }} />
                <Area type="monotone" dataKey="threshold" stroke="#e040fb" fill="url(#threshGrad)" strokeWidth={2} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
            <p className="text-xs text-[#7a8fbf] mt-2">RL mechanism: threshold rises when hit rate {'>'} 80%, falls when {'<'} 50%</p>
          </div>
        </div>
      </div>
    </div>
  );
}
