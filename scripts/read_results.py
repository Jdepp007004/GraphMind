import csv, os, json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base, "results", "benchmark_results_v2.csv")
kpi_path = os.path.join(base, "reports", "kpi_summary.json")

with open(csv_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

def g(r, k): return float(r.get(k, 0) or 0)

print(f"\n{'Policy':<24} {'HitRate':>8} {'F1':>8} {'Prec':>8} {'Lat Saved':>10} {'Battery OH':>11} {'Thrash':>8}")
print("-"*82)
for r in rows:
    print(f"{r.get('policy','?')[:24]:<24} {g(r,'cache_hit_rate'):>8.4f} {g(r,'f1'):>8.4f} "
          f"{g(r,'precision'):>8.4f} {g(r,'latency_saved_ms'):>10.1f} "
          f"{g(r,'battery_overhead_pct'):>11.4f} {g(r,'thrash_rate'):>8.4f}")

if os.path.exists(kpi_path):
    with open(kpi_path) as f:
        kpi = json.load(f)
    print("\n" + "="*60)
    print("  KPI SUMMARY")
    print("="*60)
    for k, v in kpi.items():
        if k not in ("kpi_pass_fail", "hit_at_1_pct", "static_cache_hit_rate_pct", "graphmind_vs_static_cache_improvement_pct"):
            pf = kpi["kpi_pass_fail"].get(k, "?")
            print(f"  {k:<50s}: {v}")
