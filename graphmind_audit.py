"""
GraphMind Technical Audit Script
Run from the root of the Samsung-ax-hackathon-GraphMind repo.
Prints a structured report of all detected technical gaps.
"""

import os
import ast
import re
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
REPORT = []

def section(title):
    REPORT.append(f"\n{'='*60}")
    REPORT.append(f"  {title}")
    REPORT.append(f"{'='*60}")

def finding(severity, label, detail):
    REPORT.append(f"\n[{severity}] {label}")
    REPORT.append(f"  {detail}")

def ok(label, detail=""):
    REPORT.append(f"\n[OK]  {label}" + (f" — {detail}" if detail else ""))

def read(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None

def find_files(pattern, root=ROOT):
    return list(root.rglob(pattern))

# ─── 0. REPO STRUCTURE ───────────────────────────────────────────────────────
section("0. REPO STRUCTURE")

expected_dirs = ["src", "scripts", "data", "docs", "results", "tests", "config"]
expected_files = [
    "requirements.txt", "README.md", "agents.md", ".env.example"
]

for d in expected_dirs:
    p = ROOT / d
    if p.exists():
        files = list(p.rglob("*.py"))
        ok(f"dir/{d}", f"{len(files)} .py files")
    else:
        finding("MISSING", f"dir/{d}", "Directory not found")

for f in expected_files:
    if (ROOT / f).exists():
        ok(f"file/{f}")
    else:
        finding("MISSING", f"file/{f}", "File not found")

# Print full file tree
section("0b. FULL FILE TREE")
for p in sorted(ROOT.rglob("*")):
    if any(skip in str(p) for skip in [".git", "__pycache__", ".venv", "venv", "node_modules", ".egg"]):
        continue
    rel = p.relative_to(ROOT)
    indent = "  " * (len(rel.parts) - 1)
    REPORT.append(f"  {indent}{rel.name}{'/' if p.is_dir() else ''}")

# ─── 1. ADVANCED METRICS ──────────────────────────────────────────────────────
section("1. ADVANCED METRICS — advanced_metrics.py")

metrics_files = find_files("advanced_metrics.py")
if not metrics_files:
    finding("MISSING", "advanced_metrics.py", "File not found anywhere in repo")
else:
    for mf in metrics_files:
        src = read(mf)
        REPORT.append(f"\n  File: {mf.relative_to(ROOT)}")

        # Check for hardcoded prec/rec/f1
        hardcoded_prf = re.findall(r"(prec|rec|f1|precision|recall)\s*=\s*0\.\d+", src)
        if hardcoded_prf:
            finding("CRITICAL", "Hardcoded metric constants",
                    f"Found assignments: {hardcoded_prf}")
            # Print the actual lines
            for i, line in enumerate(src.splitlines(), 1):
                if re.search(r"(prec|rec|f1|precision|recall)\s*=\s*0\.\d+", line):
                    REPORT.append(f"    line {i}: {line.strip()}")
        else:
            ok("No hardcoded prec/rec/f1 constants found")

        # Check for estimated latency reconstruction
        if "hit_rate * 0." in src or "total_events * hit_rate" in src:
            finding("CRITICAL", "Latency estimated from hit_rate, not measured",
                    "Search: 'hit_rate * 0.' or 'total_events * hit_rate'")
            for i, line in enumerate(src.splitlines(), 1):
                if "hit_rate * 0." in line or "total_events * hit_rate" in line:
                    REPORT.append(f"    line {i}: {line.strip()}")
        else:
            ok("No hit_rate-derived latency estimation found")

        # Check for real percentile computation
        if "numpy.percentile" in src or "np.percentile" in src:
            ok("np.percentile found — latency percentiles may be real")
        else:
            finding("HIGH", "np.percentile not used",
                    "P50/P95/P99 are not computed from real latency samples")

        # Check provenance tracking
        if "MEASURED" in src:
            ok("Provenance tracking (MEASURED tag) present")
        else:
            finding("MEDIUM", "No provenance tracking",
                    "MEASURED/ESTIMATED tags not found in output")

        # Dump all numeric literal assignments that look like metric values
        REPORT.append("\n  --- All suspicious constant assignments ---")
        for i, line in enumerate(src.splitlines(), 1):
            if re.search(r"=\s*0\.[3-9]\d*\b", line) and not line.strip().startswith("#"):
                REPORT.append(f"    line {i}: {line.strip()}")

# ─── 2. RL EVALUATION ─────────────────────────────────────────────────────────
section("2. RL EVALUATION — evaluation.py")

eval_files = find_files("evaluation.py")
if not eval_files:
    finding("MISSING", "evaluation.py", "Not found")
else:
    for ef in eval_files:
        src = read(ef)
        REPORT.append(f"\n  File: {ef.relative_to(ROOT)}")

        # hit_rate aliasing
        patterns = [
            ("precision = hit_rate", "precision aliased to hit_rate"),
            ("recall = hit_rate",    "recall aliased to hit_rate"),
            ("f1 = hit_rate",        "f1 aliased to hit_rate"),
        ]
        for pat, label in patterns:
            if pat in src:
                finding("CRITICAL", label, f"Found literal: '{pat}'")
                for i, line in enumerate(src.splitlines(), 1):
                    if pat in line:
                        REPORT.append(f"    line {i}: {line.strip()}")
            else:
                ok(f"Pattern '{pat}' not found")

        # Check for real TP/FP/FN
        for token in ["true_positive", "false_positive", "false_negative", "TP", "FP", "FN"]:
            if token in src:
                ok(f"Token '{token}' found in evaluation.py")
                break
        else:
            finding("CRITICAL", "No TP/FP/FN logic found",
                    "Precision/recall cannot be real without TP/FP/FN tracking")

        # Check NoOp policy
        if "NoOp" in src or "noop" in src.lower():
            for i, line in enumerate(src.splitlines(), 1):
                if "noop" in line.lower() or "NoOp" in line:
                    REPORT.append(f"  NoOp line {i}: {line.strip()}")

        # Dump full policy comparison block
        REPORT.append("\n  --- Policy names found ---")
        for token in ["Random", "NoOp", "Frequency", "LRU", "PPO", "GraphMind"]:
            if token in src:
                REPORT.append(f"    Found: {token}")

# ─── 3. THRASH RATE ───────────────────────────────────────────────────────────
section("3. THRASH RATE — memory_manager.py / policy_runner.py")

thrash_files = find_files("memory_manager.py") + find_files("graphmind_policy_runner.py") + find_files("policy_runner.py")
if not thrash_files:
    finding("MISSING", "memory_manager / policy_runner", "No relevant files found")

for tf in thrash_files:
    src = read(tf)
    REPORT.append(f"\n  File: {tf.relative_to(ROOT)}")

    # Find thrash_rate definition / increment
    thrash_lines = []
    for i, line in enumerate(src.splitlines(), 1):
        if "thrash" in line.lower():
            thrash_lines.append((i, line.strip()))

    if thrash_lines:
        REPORT.append(f"  All 'thrash' references ({len(thrash_lines)} lines):")
        for lineno, line in thrash_lines:
            REPORT.append(f"    line {lineno}: {line}")
    else:
        finding("HIGH", "No 'thrash' references found", f"in {tf.name}")

    # Check if thrash is incremented on every eviction vs. recency check
    if "thrash" in src.lower():
        has_recency = any(kw in src for kw in ["last_access", "recent", "window", "recency", "re_access"])
        if has_recency:
            ok("Recency check may exist near thrash logic")
        else:
            finding("CRITICAL", "Thrash incremented with no recency check",
                    "Every eviction likely counted as thrash. Add: 'thrash if re-accessed within N events'")

# ─── 4. DATASET GENERATOR ─────────────────────────────────────────────────────
section("4. DATASET GENERATOR")

gen_files = (find_files("dataset_generator.py") + find_files("synthetic_dataset.py")
             + find_files("persona_generator.py") + find_files("event_simulator.py"))

if not gen_files:
    finding("MISSING", "Dataset generator", "None of the expected files found")

for gf in gen_files:
    src = read(gf)
    REPORT.append(f"\n  File: {gf.relative_to(ROOT)}")

    # User count
    user_count_matches = re.findall(r"num_users\s*=\s*(\d+)|NUM_USERS\s*=\s*(\d+)|range\((\d+)\)", src)
    REPORT.append(f"  User count patterns: {user_count_matches}")

    # Persona count
    persona_count = len(re.findall(r"'persona'|\"persona\"|persona_type|PERSONA", src))
    REPORT.append(f"  'persona' references: {persona_count}")

    # Check for temporal noise / variance
    for kw in ["random.gauss", "np.random.normal", "poisson", "jitter", "noise", "variance", "std"]:
        if kw in src:
            ok(f"Temporal noise keyword found: '{kw}'")
            break
    else:
        finding("HIGH", "No temporal noise/variance found",
                "App launch times appear to be deterministic, not stochastic")

    # Check for session modeling
    if "session" in src.lower():
        ok("Session modeling keyword found")
    else:
        finding("HIGH", "No session modeling",
                "Events appear independent; real users launch in bursts/sessions")

    # Check for drift injection mid-simulation
    if "drift" in src.lower() and ("day" in src.lower() or "week" in src.lower()):
        ok("Drift injection appears to exist")
    else:
        finding("MEDIUM", "No mid-simulation drift injection",
                "Users should change habits partway through the 30-day window")

    # Check for Poisson inter-arrival
    if "poisson" in src.lower() or "exponential" in src.lower():
        ok("Poisson/exponential inter-arrival found")
    else:
        finding("MEDIUM", "No Poisson inter-arrival times",
                "Real app launches follow Poisson process, not uniform gaps")

    # Check transition matrix seeding
    if "transition" in src.lower() or "markov" in src.lower():
        ok("Transition/Markov reference found")
    else:
        finding("MEDIUM", "No explicit transition probability matrix",
                "App sequences may be fully random rather than Markov-chained")

    # Dump all numeric constants that look like user/day counts
    REPORT.append("\n  --- Numeric count constants ---")
    for i, line in enumerate(src.splitlines(), 1):
        if re.search(r"\b(10|30|100|365)\b", line) and any(
                kw in line.lower() for kw in ["user", "day", "event", "count", "num"]):
            REPORT.append(f"    line {i}: {line.strip()}")

# ─── 5. CONTEXT ENCODER ──────────────────────────────────────────────────────
section("5. CONTEXT ENCODER — context_encoder.py")

enc_files = find_files("context_encoder.py")
if not enc_files:
    finding("MISSING", "context_encoder.py", "Not found")

for ef in enc_files:
    src = read(ef)
    REPORT.append(f"\n  File: {ef.relative_to(ROOT)}")

    # Find Linear layer sizes
    linear_matches = re.findall(r"Linear\((\d+),\s*(\d+)\)", src)
    REPORT.append(f"  Linear layer sizes: {linear_matches}")

    # Find input_dim / in_features
    input_dims = re.findall(r"input_dim\s*=\s*(\d+)|in_features\s*=\s*(\d+)", src)
    REPORT.append(f"  Input dim references: {input_dims}")

    # Check for one-hot encoding
    if "one_hot" in src or "onehot" in src.lower() or "eye(" in src:
        finding("MEDIUM", "One-hot app encoding detected",
                "Will break when num_apps changes. Switch to embedding lookup or category encoding.")
    else:
        ok("No direct one-hot pattern found (may use category encoding)")

    # Check for embedding layer
    if "Embedding" in src or "nn.Embedding" in src:
        ok("nn.Embedding found — app representation is flexible")
    else:
        finding("MEDIUM", "No nn.Embedding",
                "Input size is fixed at construction time. Will require refit for different app sets.")

    # Dump full forward() signature
    REPORT.append("\n  --- forward() / encode() methods ---")
    in_forward = False
    for i, line in enumerate(src.splitlines(), 1):
        if "def forward" in line or "def encode" in line:
            in_forward = True
        if in_forward:
            REPORT.append(f"    line {i}: {line.rstrip()}")
        if in_forward and i > 0 and line.strip() == "" and i > 5:
            break

# ─── 6. GRAPH ENGINE — PRUNING ────────────────────────────────────────────────
section("6. GRAPH ENGINE — graph_engine.py / behavioural_graph.py")

graph_files = find_files("graph_engine.py") + find_files("behavioural_graph.py") + find_files("graph.py")
if not graph_files:
    finding("MISSING", "Graph engine file", "Not found")

for gf in graph_files:
    src = read(gf)
    REPORT.append(f"\n  File: {gf.relative_to(ROOT)}")

    # Pruning
    for kw in ["prune", "evict", "trim", "decay", "max_node", "MAX_NODE", "node_limit"]:
        if kw in src.lower():
            ok(f"Pruning keyword found: '{kw}'")
            for i, line in enumerate(src.splitlines(), 1):
                if kw in line.lower():
                    REPORT.append(f"    line {i}: {line.strip()}")
            break
    else:
        finding("HIGH", "No graph pruning / node limit logic",
                "Graph grows unboundedly. Add decay-based pruning triggered every N events.")

    # Edge weight decay
    if "decay" in src.lower() or "weight" in src.lower():
        ok("Edge weight or decay reference found")
    else:
        finding("MEDIUM", "No edge weight decay",
                "Old edges should decay over time to prevent stale predictions")

    # Node count bound check
    if "len(self" in src and "node" in src.lower():
        ok("Node count check may exist")

    # Dump graph class methods
    REPORT.append("\n  --- Public method names ---")
    for i, line in enumerate(src.splitlines(), 1):
        if re.match(r"\s+def [a-z]", line):
            REPORT.append(f"    line {i}: {line.strip()}")

# ─── 7. PPO / RL ENVIRONMENT ──────────────────────────────────────────────────
section("7. RL ENVIRONMENT — rl/environment.py or similar")

rl_files = (find_files("environment.py") + find_files("rl_env.py")
            + find_files("graphmind_env.py") + find_files("train_rl.py"))

if not rl_files:
    finding("MISSING", "RL environment file", "Not found")

for rf in rl_files:
    src = read(rf)
    REPORT.append(f"\n  File: {rf.relative_to(ROOT)}")

    # Reward function
    reward_lines = [(i, l.strip()) for i, l in enumerate(src.splitlines(), 1) if "reward" in l.lower()]
    REPORT.append(f"  Reward-related lines ({len(reward_lines)}):")
    for lineno, line in reward_lines[:20]:
        REPORT.append(f"    line {lineno}: {line}")

    # NoOp action
    noop_lines = [(i, l.strip()) for i, l in enumerate(src.splitlines(), 1)
                  if "noop" in l.lower() or "no_op" in l.lower() or "NoOp" in l]
    REPORT.append(f"  NoOp-related lines ({len(noop_lines)}):")
    for lineno, line in noop_lines[:10]:
        REPORT.append(f"    line {lineno}: {line}")

    # Penalty for wrong prefetch
    if "penalty" in src.lower() or "false_positive" in src.lower() or "- reward" in src:
        ok("Prefetch penalty / FP penalty found")
    else:
        finding("HIGH", "No FP penalty in reward function",
                "PPO may learn to over-prefetch everything — reward should penalize wrong prefetches")

    # Action space definition
    action_lines = [(i, l.strip()) for i, l in enumerate(src.splitlines(), 1)
                    if "action_space" in l.lower() or "ActionSpace" in l]
    REPORT.append(f"  Action space lines:")
    for lineno, line in action_lines[:5]:
        REPORT.append(f"    line {lineno}: {line}")

# ─── 8. SAMSUNG ADB / CLI ────────────────────────────────────────────────────
section("8. SAMSUNG ADB / CLI")

adb_files = (find_files("connect_samsung.py") + find_files("samsung_telemetry.py")
             + find_files("adb_collector.py") + find_files("telemetry.py"))

if not adb_files:
    finding("MISSING", "ADB/Samsung CLI files", "No ADB collector found")

for af in adb_files:
    src = read(af)
    REPORT.append(f"\n  File: {af.relative_to(ROOT)}")

    # App category mapping
    if "category" in src.lower() and ("com." in src or "package" in src.lower()):
        ok("App category mapping appears to exist")
    else:
        finding("HIGH", "No package→category mapping",
                "Raw ADB package names (com.whatsapp) need to map to categories for context encoder")

    # Calendar permission
    if "READ_CALENDAR" in src or "calendar" in src.lower():
        ok("Calendar integration found")
        if "pm grant" in src or "READ_CALENDAR" in src:
            ok("Calendar permission grant step found")
        else:
            finding("MEDIUM", "Calendar access present but no permission grant step",
                    "Add: adb shell pm grant <pkg> android.permission.READ_CALENDAR")
    else:
        finding("HIGH", "No calendar integration", "Calendar context is a key signal")

    # Multi-device support
    if "device_id" in src.lower() or "-s " in src or "serial" in src.lower():
        ok("Multi-device / device serial support found")
    else:
        finding("HIGH", "No multi-device support",
                "3 Samsung devices need unique device_id prefix in output files")

    # Polling frequency
    freq_matches = re.findall(r"sleep\((\d+\.?\d*)\)|interval\s*=\s*(\d+)", src)
    REPORT.append(f"  Polling intervals found: {freq_matches}")

    # ADB command patterns
    adb_cmds = [l.strip() for l in src.splitlines() if "adb" in l.lower() and "shell" in l.lower()]
    REPORT.append(f"  ADB shell commands ({len(adb_cmds)}):")
    for cmd in adb_cmds[:15]:
        REPORT.append(f"    {cmd}")

# ─── 9. BENCHMARK RUNNER ─────────────────────────────────────────────────────
section("9. BENCHMARK RUNNER — run_benchmarks.py")

bench_files = find_files("run_benchmarks.py")
if not bench_files:
    finding("MISSING", "run_benchmarks.py", "Not found")

for bf in bench_files:
    src = read(bf)
    REPORT.append(f"\n  File: {bf.relative_to(ROOT)}")

    # +18% boost check
    boost_patterns = ["+ 0.18", "+0.18", "+ 18", "boost", "artificial"]
    for pat in boost_patterns:
        if pat in src:
            finding("CRITICAL", f"Possible artificial boost: '{pat}'",
                    "Check if cache hit rate is being manually inflated")
            for i, line in enumerate(src.splitlines(), 1):
                if pat in line:
                    REPORT.append(f"    line {i}: {line.strip()}")

    # Check baselines
    for baseline in ["LMKD", "Bixby", "UsageStats", "ART", "LRU", "Frequency"]:
        if baseline in src:
            ok(f"Baseline '{baseline}' found")
        else:
            finding("LOW", f"Baseline '{baseline}' not found", "")

    # Provenance
    if "MEASURED" in src and "ESTIMATED" in src:
        ok("MEASURED/ESTIMATED provenance tagging found")
    else:
        finding("MEDIUM", "No MEASURED/ESTIMATED provenance tagging in benchmark output", "Add MEASURED/ESTIMATED labels to benchmark JSON output")

# ─── 10. SCALE TEST ──────────────────────────────────────────────────────────
section("10. SCALE TEST — run_scale_test.py")

scale_files = find_files("run_scale_test.py") + find_files("scale_test.py")
if not scale_files:
    finding("MISSING", "Scale test file", "Not found")

for sf in scale_files:
    src = read(sf)
    REPORT.append(f"\n  File: {sf.relative_to(ROOT)}")

    # Verify independent graphs per user
    if "BehaviouralGraph(user_id)" in src or "BehavioralGraph(user_id)" in src or "Graph(user" in src:
        ok("Independent graph per user confirmed")
    else:
        finding("HIGH", "Cannot confirm independent graph per user",
                "Scale test may be reusing a single graph instance — check instantiation")

    scale_counts = re.findall(r"\b(10|100|1000|10000)\b", src)
    REPORT.append(f"  Scale levels found: {scale_counts}")

# ─── 11. RESULTS / DATA FILES ────────────────────────────────────────────────
section("11. RESULTS AND DATA FILES")

results_dir = ROOT / "results"
data_dir = ROOT / "data"

for d, label in [(results_dir, "results/"), (data_dir, "data/")]:
    if d.exists():
        files = list(d.rglob("*"))
        REPORT.append(f"\n  {label} — {len(files)} total entries:")
        for f in sorted(files)[:40]:
            if f.is_file():
                size = f.stat().st_size
                REPORT.append(f"    {f.relative_to(ROOT)} ({size} bytes)")
    else:
        finding("MISSING", label, "Directory not found")

# Spot-check a results JSON
json_files = list(results_dir.rglob("*.json")) if results_dir.exists() else []
for jf in json_files[:3]:
    try:
        data = json.loads(jf.read_text(encoding="utf-8"))
        REPORT.append(f"\n  Sample JSON keys in {jf.name}: {list(data.keys())[:10]}")
        # Check for constant values across users
        if isinstance(data, list) and len(data) > 1:
            sample_keys = list(data[0].keys()) if isinstance(data[0], dict) else []
            for k in sample_keys:
                vals = [row.get(k) for row in data if isinstance(row, dict)]
                if len(set(str(v) for v in vals)) == 1:
                    finding("HIGH", f"Constant value across all rows: '{k}'",
                            f"Value is always: {vals[0]}")
    except Exception as e:
        REPORT.append(f"  Could not parse {jf.name}: {e}")

# ─── 12. REQUIREMENTS & DEPENDENCIES ─────────────────────────────────────────
section("12. REQUIREMENTS.TXT")

req = read(ROOT / "requirements.txt")
if req:
    REPORT.append("\n  Packages listed:")
    for line in req.splitlines():
        if line.strip() and not line.startswith("#"):
            REPORT.append(f"    {line.strip()}")

    # Check for key packages
    for pkg in ["torch", "stable-baselines3", "streamlit", "networkx", "numpy", "pandas", "scipy"]:
        if pkg in req:
            ok(f"Package '{pkg}' in requirements")
        else:
            finding("LOW", f"Package '{pkg}' not in requirements.txt",
                    "May be missing or named differently")
else:
    finding("MISSING", "requirements.txt", "")

# ─── 13. AGENTS.MD ───────────────────────────────────────────────────────────
section("13. AGENTS.MD AND DOCS")

agents_md = read(ROOT / "agents.md")
if agents_md:
    word_count = len(agents_md.split())
    REPORT.append(f"\n  agents.md found — {word_count} words")
    for kw in ["GraphManagerAgent", "DriftDetectorAgent", "RLTrainerAgent",
               "PrefetchAgent", "SecurityAgent", "LangGraph", "tool"]:
        if kw in agents_md:
            ok(f"Keyword '{kw}' in agents.md")
        else:
            finding("LOW", f"'{kw}' not found in agents.md", "Add this to agents.md for completeness")
else:
    finding("MISSING", "agents.md", "")

docs_dir = ROOT / "docs"
if docs_dir.exists():
    doc_files = list(docs_dir.rglob("*.md"))
    REPORT.append(f"\n  docs/ contains {len(doc_files)} markdown files:")
    for df in doc_files:
        REPORT.append(f"    {df.relative_to(ROOT)}")
    if not any("ax" in f.name.lower() for f in doc_files):
        finding("HIGH", "docs/ax.md missing",
                "Required by submission template — explain agentic AI setup here")
else:
    finding("MISSING", "docs/", "Directory not found")

# ─── 14. DASHBOARD ───────────────────────────────────────────────────────────
section("14. DASHBOARD — src/dashboard/app.py")

dash_files = find_files("app.py") + find_files("dashboard.py")
if not dash_files:
    finding("MISSING", "Dashboard file", "")
else:
    for df in dash_files:
        src = read(df)
        REPORT.append(f"\n  File: {df.relative_to(ROOT)}")
        tabs = re.findall(r"st\.tab|tab\[|Tab\(", src)
        REPORT.append(f"  Tab widgets found: {len(tabs)}")
        for kw in ["thrash", "precision", "recall", "drift", "security", "explainab"]:
            if kw in src.lower():
                ok(f"Dashboard shows '{kw}' data")
            else:
                finding("LOW", f"Dashboard missing '{kw}' visualization", "Add a tab or section for this metric")

# ─── FINAL SUMMARY ───────────────────────────────────────────────────────────
section("SUMMARY — Issue Counts")

critical = sum(1 for l in REPORT if l.startswith("[CRITICAL]"))
high     = sum(1 for l in REPORT if l.startswith("[HIGH]"))
medium   = sum(1 for l in REPORT if l.startswith("[MEDIUM]"))
low      = sum(1 for l in REPORT if l.startswith("[LOW]"))
missing  = sum(1 for l in REPORT if l.startswith("[MISSING]"))

REPORT.append(f"\n  CRITICAL : {critical}")
REPORT.append(f"  HIGH     : {high}")
REPORT.append(f"  MEDIUM   : {medium}")
REPORT.append(f"  LOW      : {low}")
REPORT.append(f"  MISSING  : {missing}")
REPORT.append(f"\n  Total issues: {critical + high + medium + low + missing}")

# ─── PRINT ───────────────────────────────────────────────────────────────────
output = "\n".join(REPORT)
print(output)

out_path = ROOT / "AUDIT_REPORT.txt"
out_path.write_text(output, encoding="utf-8")
print(f"\n\nReport also saved to: {out_path}")