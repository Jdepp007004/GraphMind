#!/usr/bin/env python3
"""
scripts/generate_dashboard_data.py

Converts all Python/CSV/Parquet data sources into JSON files
consumed by the Next.js dashboard.

Outputs (dashboard/public/data/):
  benchmark.json          - full policy comparison table
  optimization.json       - Phase 11 optimization journey
  weight_grid.json        - Phase A weight grid results
  threshold_sweep.json    - Phase B threshold sweep results
  users.json              - user metadata
  graph.json              - app transition graph (top user)
  transitions.json        - sample transition sequences for playback
  ablations.json          - ablation study results
  latency.json            - latency measurement data
"""

import csv, json, os, sys
from collections import defaultdict
from typing import Dict, List, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUT_DIR = os.path.join(PROJECT_ROOT, "dashboard", "public", "data")
RESULTS = os.path.join(PROJECT_ROOT, "results")
REPORTS = os.path.join(PROJECT_ROOT, "reports")
PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
DATASETS = os.path.join(PROJECT_ROOT, "datasets")
UBIQLOG = os.path.join(DATASETS, "ubiqlog", "UbiqLog4UCI")

os.makedirs(OUT_DIR, exist_ok=True)


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(name, data):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    size = os.path.getsize(path)
    print(f"  [OK] {name} ({size//1024}KB, {len(data) if isinstance(data, list) else 'dict'} items)")


def coerce(v, typ=float, default=0.0):
    try: return typ(v)
    except: return default


# ── 1. Benchmark Results ─────────────────────────────────────────────────────
def gen_benchmark():
    rows = read_csv(os.path.join(RESULTS, "final_production_results.csv"))
    order = [
        "GraphMindRL_V5", "GraphMindRL_V5_t10", "RL_LatencyFocus",
        "GraphMindRL_Baseline", "Graph+Confidence",
        "Markov2", "Markov1", "GraphOnly", "GlobalMarkov2",
    ]
    ordered = sorted(rows, key=lambda r: order.index(r["policy"]) if r["policy"] in order else 99)
    out = []
    for r in ordered:
        out.append({
            "policy":       r["policy"],
            "f1":           coerce(r["f1"]),
            "std_f1":       coerce(r.get("std_f1", 0)),
            "precision":    coerce(r.get("precision", 0)),
            "recall":       coerce(r.get("recall", 0)),
            "hit_rate":     coerce(r.get("hit_rate", 0)),
            "latency_saved_ms": coerce(r.get("latency_saved_ms", 0)),
            "delta_f1":     coerce(r.get("delta_f1_vs_baseline", 0)),
            "p_value":      coerce(r.get("p_value", 1.0)),
            "cohen_d":      coerce(r.get("cohen_d", 0)),
            "significant":  r.get("significant", "False").lower() == "true",
            "n_users":      coerce(r.get("n_users", 31), int, 31),
            "config":       r.get("config", ""),
        })
    write_json("benchmark.json", out)
    return out


# ── 2. Optimization Journey ──────────────────────────────────────────────────
def gen_optimization():
    timeline = [
        {
            "phase": "Start", "label": "GraphOnly (= Markov-1)",
            "f1": 0.7267, "delta": 0.0, "status": "baseline",
            "description": "Architecture audit proves GraphOnly is mathematically identical to Markov-1. P(next|current) with top-k selection.",
            "date": "2026-05-15", "result": "baseline",
        },
        {
            "phase": "Phase 3–4", "label": "Markov-2 Order Analysis",
            "f1": 0.7295, "delta": 0.0028, "status": "weak",
            "description": "Bigram model P(next|prev,curr) tested. Only +0.003 F1, p>0.05. Not statistically significant on 31 users.",
            "date": "2026-05-20", "result": "rejected",
        },
        {
            "phase": "Phase 5–6", "label": "Time Context Evaluation",
            "f1": 0.7241, "delta": -0.0026, "status": "failed",
            "description": "TimeAwareM1 with 6/12/24/48 time bands. Phase 11C audit: 98.5% coverage but signal quality reduces F1. Excluded from scoring.",
            "date": "2026-05-25", "result": "rejected",
        },
        {
            "phase": "Phase 7", "label": "RL as Threshold Controller",
            "f1": 0.7539, "delta": 0.0115, "status": "accepted",
            "description": "RL agent adapts prefetch threshold ±0.005 based on 20-step rolling hit rate. First statistically significant improvement. p=0.0003, d=0.752.",
            "date": "2026-05-30", "result": "accepted",
        },
        {
            "phase": "Phase 11D", "label": "Modified Kneser-Ney",
            "f1": 0.7276, "delta": -0.0148, "status": "failed",
            "description": "P_MKN = λ2×P(C|A,B) + (1-λ2)×P(C|B). All K values (3/5/10) below baseline. Bigram without unigram fallback is worse.",
            "date": "2026-06-02", "result": "rejected",
        },
        {
            "phase": "Phase 11A", "label": "Confidence Weight Grid Search",
            "f1": 0.7733, "delta": 0.0309, "status": "accepted",
            "description": "Grid over trans/rec/freq weights. Key insight: frequency=0.4 (was 0.2) captures habitual app usage. Recency=0.1 (was 0.3) was overweighted.",
            "date": "2026-06-06", "result": "accepted",
        },
        {
            "phase": "Phase 11B", "label": "Threshold Optimization",
            "f1": 0.7564, "delta": 0.0140, "status": "accepted",
            "description": "Sweep threshold 0.02–0.20 with baseline weights. Best: 0.16. Combined with weight optimization for Phase E.",
            "date": "2026-06-06", "result": "accepted",
        },
        {
            "phase": "Phase 11E", "label": "GraphMindRL_V5 Combined",
            "f1": 0.7745, "delta": 0.0321, "status": "production",
            "description": "Best weights (0.5/0.1/0.4) + best threshold (0.16) + adaptive RL mechanism. F1=0.7745, p=0.0115, d=0.491. Meets both success criteria.",
            "date": "2026-06-06", "result": "production",
        },
    ]
    write_json("optimization.json", timeline)


# ── 3. Weight Grid ────────────────────────────────────────────────────────────
def gen_weight_grid():
    rows = read_csv(os.path.join(RESULTS, "v5_weight_grid.csv"))
    out = []
    for r in rows:
        out.append({
            "weights": r.get("weights", ""),
            "w_trans": coerce(r.get("w_trans", 0)),
            "w_rec":   coerce(r.get("w_rec", 0)),
            "w_freq":  coerce(r.get("w_freq", 0)),
            "f1":      coerce(r.get("f1", 0)),
            "std_f1":  coerce(r.get("std_f1", 0)),
            "delta_f1": coerce(r.get("delta_f1", 0)),
            "hit_rate": coerce(r.get("hit_rate", 0)),
            "latency_saved": coerce(r.get("latency_saved", 0)),
        })
    out.sort(key=lambda x: -x["f1"])
    write_json("weight_grid.json", out)


# ── 4. Threshold Sweep ────────────────────────────────────────────────────────
def gen_threshold_sweep():
    rows = read_csv(os.path.join(RESULTS, "v5_threshold_sweep.csv"))
    out = []
    for r in rows:
        out.append({
            "threshold":    coerce(r.get("threshold", 0)),
            "f1":           coerce(r.get("f1", 0)),
            "precision":    coerce(r.get("precision", 0)),
            "recall":       coerce(r.get("recall", 0)),
            "hit_rate":     coerce(r.get("hit_rate", 0)),
            "delta_f1":     coerce(r.get("delta_f1", 0)),
        })
    out.sort(key=lambda x: x["threshold"])
    write_json("threshold_sweep.json", out)


# ── 5. Users ─────────────────────────────────────────────────────────────────
def gen_users():
    path = os.path.join(PROCESSED, "users.json")
    if not os.path.exists(path):
        print("  ⚠ users.json not found")
        return
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    users = data.get("users", data) if isinstance(data, dict) else data
    # Summarize for dashboard
    out = []
    for u in users[:31]:  # top 31 usable
        out.append({
            "user_id": u.get("user_id", ""),
            "n_events": coerce(u.get("n_events", u.get("n_apps", 0)), int, 0),
            "n_transitions": coerce(u.get("n_transitions", 0), int, 0),
            "n_unique_apps": coerce(u.get("n_unique_apps", 0), int, 0),
            "gender": u.get("gender", u.get("user_id", "")[-1] if u.get("user_id", "") else ""),
            "date_start": u.get("date_start", u.get("first_date", "")),
            "date_end": u.get("date_end", u.get("last_date", "")),
        })
    write_json("users.json", out)


# ── 6. App Transition Graph ───────────────────────────────────────────────────
def gen_graph():
    """Build a Markov graph from raw transitions for the most active user."""
    import json as js

    # Find best user from users.json
    users_path = os.path.join(PROCESSED, "users.json")
    best_uid = None
    if os.path.exists(users_path):
        with open(users_path, encoding="utf-8") as f:
            data = js.load(f)
        users = data.get("users", data) if isinstance(data, dict) else data
        if users:
            # Pick user with most transitions
            best = max(users[:31], key=lambda u: coerce(u.get("n_transitions", 0), int, 0))
            best_uid = best.get("user_id")

    if not best_uid:
        best_uid = "8_M"  # fallback to known good user

    # Load user raw data
    user_dir = os.path.join(UBIQLOG, best_uid)
    if not os.path.exists(user_dir):
        # Try first available
        available = [d for d in os.listdir(UBIQLOG) if os.path.isdir(os.path.join(UBIQLOG, d))]
        best_uid = available[0] if available else None

    if not best_uid:
        print("  ⚠ No user data found for graph generation")
        return

    # Parse transitions
    import datetime
    SYSTEM_PREFIXES = ("com.android.", "com.google.android.providers",
                       "com.google.android.gms", "com.google.android.gsf",
                       "com.sec.android.provider", "com.samsung.android.provider",
                       "com.redbend.", "android.")

    raw = []
    user_dir = os.path.join(UBIQLOG, best_uid)
    for fname in sorted(os.listdir(user_dir)):
        if not fname.endswith(".txt"):
            continue
        try:
            with open(os.path.join(user_dir, fname), encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = js.loads(line)
                        if "Application" not in obj:
                            continue
                        app = obj["Application"]
                        pkg = app.get("ProcessName", "").strip()
                        if not pkg:
                            continue
                        if any(pkg.lower().startswith(p) for p in SYSTEM_PREFIXES):
                            continue
                        ts_str = app.get("Start", "")
                        try:
                            dt = datetime.datetime.strptime(ts_str.strip(), "%m-%d-%Y %H:%M:%S")
                            if not (2011 <= dt.year <= 2016):
                                continue
                        except:
                            continue
                        raw.append((dt, pkg))
                    except:
                        pass
        except:
            pass

    raw.sort(key=lambda x: x[0])
    apps = [r[1] for r in raw]

    # Build transition counts
    edges = defaultdict(lambda: defaultdict(int))
    for i in range(1, min(len(apps), 3000)):  # use first 3000 transitions
        if (raw[i][0] - raw[i-1][0]).total_seconds() <= 3600:
            edges[apps[i-1]][apps[i]] += 1

    # Get top apps by degree
    node_freq = defaultdict(int)
    for src, dsts in edges.items():
        for dst, cnt in dsts.items():
            node_freq[src] += cnt
            node_freq[dst] += cnt

    top_apps = sorted(node_freq, key=lambda a: -node_freq[a])[:30]
    top_set = set(top_apps)

    # Shorten package names for display
    def short_name(pkg):
        parts = pkg.split(".")
        return parts[-1] if parts else pkg

    # Build nodes
    nodes = []
    for i, app in enumerate(top_apps):
        out_deg = sum(edges[app][dst] for dst in edges[app] if dst in top_set)
        nodes.append({
            "id": app,
            "label": short_name(app),
            "full_pkg": app,
            "frequency": node_freq[app],
            "out_degree": out_deg,
        })

    # Build edges
    edge_list = []
    for src in top_set:
        total_out = sum(edges[src][dst] for dst in edges[src])
        if total_out == 0:
            continue
        for dst in top_set:
            cnt = edges[src].get(dst, 0)
            if cnt > 0:
                prob = cnt / total_out
                if prob >= 0.05:  # only edges >= 5% probability
                    edge_list.append({
                        "source": src,
                        "target": dst,
                        "count": cnt,
                        "probability": round(prob, 4),
                        "label": f"{prob*100:.0f}%",
                    })

    edge_list.sort(key=lambda e: -e["probability"])

    graph_data = {
        "user_id": best_uid,
        "n_total_apps": len(apps),
        "n_transitions_used": min(len(apps)-1, 2999),
        "nodes": nodes,
        "edges": edge_list[:200],  # cap at 200 edges
    }
    write_json("graph.json", graph_data)


# ── 7. Transition Playback Data ───────────────────────────────────────────────
def gen_transitions():
    """Sample transition sequence for cache simulator / playback."""
    import json as js, datetime

    SYSTEM_PREFIXES = ("com.android.", "com.google.android.providers",
                       "com.google.android.gms", "com.google.android.gsf",
                       "com.sec.android.provider", "com.samsung.android.provider",
                       "com.redbend.", "android.")

    # Load multiple users for playback options
    users_path = os.path.join(PROCESSED, "users.json")
    uids = []
    if os.path.exists(users_path):
        with open(users_path, encoding="utf-8") as f:
            data = js.load(f)
        users = data.get("users", data) if isinstance(data, dict) else data
        uids = [u["user_id"] for u in sorted(users[:31],
                key=lambda u: coerce(u.get("n_transitions", 0), int, 0), reverse=True)[:5]]
    if not uids:
        uids = ["8_M", "14_F", "24_F"]

    all_users_data = {}
    for uid in uids:
        user_dir = os.path.join(UBIQLOG, uid)
        if not os.path.exists(user_dir):
            continue

        raw = []
        for fname in sorted(os.listdir(user_dir)):
            if not fname.endswith(".txt"):
                continue
            try:
                with open(os.path.join(user_dir, fname), encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = js.loads(line)
                            if "Application" not in obj:
                                continue
                            app = obj["Application"]
                            pkg = app.get("ProcessName", "").strip()
                            if not pkg or any(pkg.lower().startswith(p) for p in SYSTEM_PREFIXES):
                                continue
                            ts_str = app.get("Start", "")
                            try:
                                dt = datetime.datetime.strptime(ts_str.strip(), "%m-%d-%Y %H:%M:%S")
                                if not (2011 <= dt.year <= 2016):
                                    continue
                            except:
                                continue
                            raw.append((dt, pkg))
                        except:
                            pass
            except:
                pass

        raw.sort(key=lambda x: x[0])

        # Take test portion (last 10%)
        n = len(raw)
        test_start = int(n * 0.90)
        test_seq = raw[test_start:test_start + 200]  # 200 events for playback

        if len(test_seq) < 10:
            continue

        # Simulate GraphMindRL_V5 predictions on this sequence
        # Build simple Markov from training portion
        train = raw[:int(n * 0.80)]
        counts = defaultdict(lambda: defaultdict(int))
        freq = defaultdict(int)
        for i in range(1, len(train)):
            if (train[i][0] - train[i-1][0]).total_seconds() <= 3600:
                counts[train[i-1][1]][train[i][1]] += 1
                freq[train[i][1]] += 1

        total_freq = sum(freq.values()) or 1

        def short(pkg):
            return pkg.split(".")[-1]

        events = []
        hot_cache = []
        warm_cache = []
        recency = defaultdict(float)
        total_events = 0
        threshold = 0.16
        hit_history = []

        for i, (dt, pkg) in enumerate(test_seq):
            total_events += 1
            for k in recency: recency[k] *= 0.95
            recency[pkg] = 1.0
            freq[pkg] += 1

            # Compute predictions
            preds = []
            if pkg in counts:
                tot_out = sum(counts[pkg].values())
                max_rec = max(recency.values()) if recency else 1.0
                for nxt, cnt in sorted(counts[pkg].items(), key=lambda x: -x[1])[:20]:
                    trans_p = cnt / tot_out
                    rec_p = recency.get(nxt, 0) / max(max_rec, 1e-9)
                    freq_p = freq.get(nxt, 0) / total_freq
                    confidence = 0.5*trans_p + 0.1*rec_p + 0.4*freq_p
                    if confidence >= threshold:
                        preds.append({
                            "app": nxt,
                            "short": short(nxt),
                            "confidence": round(confidence, 4),
                            "trans_prob": round(trans_p, 4),
                        })
                preds = preds[:5]

            # Cache lookup
            tier = "hot" if pkg in hot_cache else ("warm" if pkg in warm_cache else "miss")
            is_hit = tier != "miss"

            # Latency saved
            lat = {
                "hot":  max(0, 2763 - 274),
                "warm": max(0, 2763 - 1301),
                "miss": 0,
            }[tier]

            # Adaptive threshold
            hit_history.append(1.0 if is_hit else 0.0)
            if len(hit_history) > 20: hit_history.pop(0)
            if len(hit_history) == 20:
                hr = sum(hit_history) / 20
                if hr < 0.5:   threshold = max(0.05, threshold - 0.005)
                elif hr > 0.8: threshold = min(0.25, threshold + 0.005)

            # Update cache
            if pkg in hot_cache:  hot_cache.remove(pkg)
            elif pkg in warm_cache: warm_cache.remove(pkg)
            hot_cache.insert(0, pkg)
            while len(hot_cache) > 5:
                warm_cache.insert(0, hot_cache.pop())
            while len(warm_cache) > 15:
                warm_cache.pop()

            # Prefetch predictions
            for p in preds:
                a = p["app"]
                if a not in hot_cache and a not in warm_cache:
                    warm_cache.insert(0, a)
                    while len(warm_cache) > 15:
                        warm_cache.pop()

            events.append({
                "step":          i,
                "timestamp":     dt.strftime("%Y-%m-%d %H:%M"),
                "app":           pkg,
                "short":         short(pkg),
                "tier":          tier,
                "hit":           is_hit,
                "latency_saved": lat,
                "threshold":     round(threshold, 3),
                "predictions":   preds,
                "hot_cache":     [short(a) for a in hot_cache],
                "warm_cache":    [short(a) for a in warm_cache[:8]],  # show first 8
            })

        all_users_data[uid] = {
            "user_id": uid,
            "n_events": len(events),
            "events":   events,
        }

    write_json("transitions.json", all_users_data)


# ── 8. Ablation Summary ───────────────────────────────────────────────────────
def gen_ablations():
    # Key ablation findings from the research
    ablations = [
        {"component": "Full GraphMindRL_V5", "f1": 0.7745, "delta": 0.0, "note": "Production model"},
        {"component": "- Frequency weight (freq=0.2)", "f1": 0.7550, "delta": -0.0195, "note": "Remove freq boost"},
        {"component": "- Adaptive threshold (fixed=0.16)", "f1": 0.7564, "delta": -0.0181, "note": "No RL adaptation"},
        {"component": "- RL (fixed threshold=0.05)", "f1": 0.7539, "delta": -0.0206, "note": "Baseline config"},
        {"component": "Graph only (= Markov-1)", "f1": 0.7267, "delta": -0.0478, "note": "No confidence layer"},
        {"component": "Time context added (6-band)", "f1": 0.7241, "delta": -0.0504, "note": "Hurts on short data"},
        {"component": "Modified KN (K=5)", "f1": 0.7276, "delta": -0.0469, "note": "Bigram smoothing"},
        {"component": "Global Markov-2", "f1": 0.6790, "delta": -0.0955, "note": "Cross-user transfer"},
    ]
    write_json("ablations.json", ablations)


# ── 9. Latency Data ───────────────────────────────────────────────────────────
def gen_latency():
    path = os.path.join(DATASETS, "app_launch_latency.csv")
    if not os.path.exists(path):
        print("  ⚠ latency CSV not found")
        return
    rows = read_csv(path)
    # Aggregate by app + start_type
    agg = defaultdict(list)
    for r in rows:
        key = (r.get("app_id", r.get("package_name", "")), r.get("start_type", "cold"))
        agg[key].append(coerce(r.get("total_time_ms", 0)))

    result = []
    by_app = defaultdict(dict)
    for (aid, tier), vals in agg.items():
        by_app[aid][tier] = round(sum(vals)/len(vals), 1)

    for aid, tiers in sorted(by_app.items()):
        short = aid.split(".")[-1] if "." in aid else aid
        result.append({
            "app_id": aid,
            "short_name": short,
            "cold_ms": tiers.get("cold", 2763),
            "warm_ms": tiers.get("warm", 1301),
            "hot_ms":  tiers.get("hot", 274),
            "saved_hot_ms":  round(tiers.get("cold", 2763) - tiers.get("hot", 274), 1),
            "saved_warm_ms": round(tiers.get("cold", 2763) - tiers.get("warm", 1301), 1),
        })
    write_json("latency.json", result)


# ── Summary stats ─────────────────────────────────────────────────────────────
def gen_summary():
    summary = {
        "model": "GraphMindRL_V5",
        "f1": 0.7745,
        "delta_f1": 0.0321,
        "baseline_f1": 0.7424,
        "hit_rate": 0.9307,
        "latency_saved_ms": 1847,
        "p_value": 0.0115,
        "cohen_d": 0.491,
        "n_users": 31,
        "n_transitions": 208695,
        "n_events": 820603,
        "n_unique_apps": "~280 per user",
        "dataset": "UbiqLog4UCI",
        "split": "80/10/10 chronological",
        "config": {
            "w_transition": 0.50,
            "w_recency": 0.10,
            "w_frequency": 0.40,
            "w_context": 0.00,
            "threshold": 0.16,
            "threshold_adapt_step": 0.005,
            "hot_size": 5,
            "warm_size": 15,
        },
        "cold_start_ms": 2763,
        "warm_start_ms": 1301,
        "hot_start_ms": 274,
        "device": "Samsung Galaxy A23",
        "measurements": 3900,
    }
    write_json("summary.json", summary)


if __name__ == "__main__":
    print("Generating dashboard data...")
    gen_summary()
    gen_benchmark()
    gen_optimization()
    gen_weight_grid()
    gen_threshold_sweep()
    gen_users()
    print("Building app transition graph (may take a moment)...")
    gen_graph()
    print("Building transition playback data (may take a moment)...")
    gen_transitions()
    gen_ablations()
    gen_latency()
    print(f"\nAll data written to: {OUT_DIR}")
