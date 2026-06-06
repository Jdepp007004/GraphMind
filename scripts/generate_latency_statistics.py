#!/usr/bin/env python3
"""
scripts/generate_latency_statistics.py

Phase 4: Generate latency statistics from measured Samsung Galaxy A23 data.

Source: datasets/app_launch_latency.csv
  - 3,900 rows (13 apps × 3 tiers × 100 samples)
  - Tiers: cold, warm, hot
  - Columns: timestamp, app_id, package_name, category, start_type,
             this_time_ms, total_time_ms, wait_time_ms, ...

Outputs:
  - reports/latency_statistics.csv
  - reports/latency_statistics.md
"""

import csv
import logging
import os
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LATENCY_CSV  = os.path.join(PROJECT_ROOT, "datasets", "app_launch_latency.csv")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    with open(LATENCY_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    logger.info(f"Loaded {len(rows)} rows from {LATENCY_CSV}")

    # Group by (app_id, start_type)
    groups: dict = {}
    meta: dict = {}
    for r in rows:
        app_id     = r["app_id"]
        start_type = r["start_type"]
        key = (app_id, start_type)
        if key not in groups:
            groups[key] = []
        try:
            groups[key].append(float(r["total_time_ms"]))
        except ValueError:
            pass
        if app_id not in meta:
            meta[app_id] = {
                "package_name": r.get("package_name", ""),
                "category":     r.get("category", ""),
            }

    # Compute statistics
    stats_rows = []
    for (app_id, start_type), values in sorted(groups.items()):
        arr = np.array(values)
        stats_rows.append({
            "app_id":       app_id,
            "package_name": meta[app_id]["package_name"],
            "category":     meta[app_id]["category"],
            "start_type":   start_type,
            "n":            int(len(arr)),
            "mean_ms":      round(float(arr.mean()),                1),
            "median_ms":    round(float(np.median(arr)),            1),
            "p50_ms":       round(float(np.percentile(arr, 50)),    1),
            "p90_ms":       round(float(np.percentile(arr, 90)),    1),
            "p95_ms":       round(float(np.percentile(arr, 95)),    1),
            "p99_ms":       round(float(np.percentile(arr, 99)),    1),
            "std_ms":       round(float(arr.std()),                 1),
            "min_ms":       round(float(arr.min()),                 1),
            "max_ms":       round(float(arr.max()),                 1),
        })

    # Write CSV
    csv_path = os.path.join(REPORTS_DIR, "latency_statistics.csv")
    fieldnames = [
        "app_id", "package_name", "category", "start_type", "n",
        "mean_ms", "median_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms",
        "std_ms", "min_ms", "max_ms"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(stats_rows)
    logger.info(f"Written: {csv_path}")

    # Compute latency saved (cold - warm, cold - hot)
    cold = {r["app_id"]: r["mean_ms"] for r in stats_rows if r["start_type"] == "cold"}
    warm = {r["app_id"]: r["mean_ms"] for r in stats_rows if r["start_type"] == "warm"}
    hot  = {r["app_id"]: r["mean_ms"] for r in stats_rows if r["start_type"] == "hot"}

    # Write markdown report
    md_path = os.path.join(REPORTS_DIR, "latency_statistics.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Galaxy A23 App Launch Latency Statistics\n\n")
        f.write("**Source:** `datasets/app_launch_latency.csv`  \n")
        f.write("**Device:** Samsung Galaxy A23 (SM-A235F/DS, Android 14, OneUI 6.1)  \n")
        f.write("**Measurement:** ADB `am start -W`, 100 launches per app per tier  \n")
        f.write(f"**Total rows:** {len(rows):,} (13 apps × 3 tiers × 100 samples)  \n\n")
        f.write("---\n\n")

        f.write("## Cold Start Latency (ms)\n\n")
        f.write("| App | Package | Category | Mean | Median | P90 | P95 | P99 | Std |\n")
        f.write("|-----|---------|----------|------|--------|-----|-----|-----|-----|\n")
        for r in sorted(stats_rows, key=lambda x: -x["mean_ms"]):
            if r["start_type"] != "cold":
                continue
            f.write(
                f"| {r['app_id']} | `{r['package_name']}` | {r['category']} "
                f"| {r['mean_ms']:.0f} | {r['median_ms']:.0f} "
                f"| {r['p90_ms']:.0f} | {r['p95_ms']:.0f} "
                f"| {r['p99_ms']:.0f} | {r['std_ms']:.0f} |\n"
            )

        f.write("\n## Warm Start Latency (ms)\n\n")
        f.write("| App | Mean | Median | P90 | P95 | P99 | Std | Saved vs Cold |\n")
        f.write("|-----|------|--------|-----|-----|-----|-----|---------------|\n")
        for r in sorted(stats_rows, key=lambda x: -x["mean_ms"]):
            if r["start_type"] != "warm":
                continue
            saved = cold.get(r["app_id"], 0) - r["mean_ms"]
            f.write(
                f"| {r['app_id']} | {r['mean_ms']:.0f} | {r['median_ms']:.0f} "
                f"| {r['p90_ms']:.0f} | {r['p95_ms']:.0f} "
                f"| {r['p99_ms']:.0f} | {r['std_ms']:.0f} "
                f"| **{saved:.0f} ms** |\n"
            )

        f.write("\n## Hot Start Latency (ms)\n\n")
        f.write("| App | Mean | Median | P90 | P95 | P99 | Std | Saved vs Cold |\n")
        f.write("|-----|------|--------|-----|-----|-----|-----|---------------|\n")
        for r in sorted(stats_rows, key=lambda x: -x["mean_ms"]):
            if r["start_type"] != "hot":
                continue
            saved = cold.get(r["app_id"], 0) - r["mean_ms"]
            f.write(
                f"| {r['app_id']} | {r['mean_ms']:.0f} | {r['median_ms']:.0f} "
                f"| {r['p90_ms']:.0f} | {r['p95_ms']:.0f} "
                f"| {r['p99_ms']:.0f} | {r['std_ms']:.0f} "
                f"| **{saved:.0f} ms** |\n"
            )

        # Summary table
        f.write("\n## Latency Savings Summary\n\n")
        f.write("| App | Cold (ms) | Warm (ms) | Hot (ms) | Warm Saves | Hot Saves | Hot Saves % |\n")
        f.write("|-----|-----------|-----------|----------|-----------|-----------|-------------|\n")
        apps_sorted = sorted(cold.keys(), key=lambda a: -cold[a])
        for app_id in apps_sorted:
            c = cold.get(app_id, 0)
            w = warm.get(app_id, 0)
            h = hot.get(app_id, 0)
            w_saved = c - w
            h_saved = c - h
            h_pct   = (h_saved / c * 100) if c > 0 else 0
            f.write(
                f"| {app_id} | {c:.0f} | {w:.0f} | {h:.0f} "
                f"| {w_saved:.0f} ms | {h_saved:.0f} ms "
                f"| {h_pct:.1f}% |\n"
            )

        # Overall averages
        all_cold = [cold[a] for a in cold]
        all_warm = [warm[a] for a in warm]
        all_hot  = [hot[a]  for a in hot]
        avg_c = np.mean(all_cold)
        avg_w = np.mean(all_warm)
        avg_h = np.mean(all_hot)
        f.write(f"\n**Average cold start:** {avg_c:.0f} ms  \n")
        f.write(f"**Average warm start:** {avg_w:.0f} ms  \n")
        f.write(f"**Average hot start:** {avg_h:.0f} ms  \n")
        f.write(f"**Average warm saving:** {avg_c - avg_w:.0f} ms ({(avg_c - avg_w)/avg_c*100:.1f}%)  \n")
        f.write(f"**Average hot saving:** {avg_c - avg_h:.0f} ms ({(avg_c - avg_h)/avg_c*100:.1f}%)  \n")

    logger.info(f"Written: {md_path}")
    logger.info(f"Average cold: {avg_c:.0f}ms  warm: {avg_w:.0f}ms  hot: {avg_h:.0f}ms")


if __name__ == "__main__":
    main()
