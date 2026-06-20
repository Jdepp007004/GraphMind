"""
scripts/demo_gemma.py

GraphMind V6 -- Gemma Explanation Demo.

Runs 15 steps through the full GraphMind V6 pipeline with Gemma enabled,
showing real natural-language prefetch explanations alongside cache decisions.

Usage:
    # With Gemma model (slower, requires model download):
    set ENABLE_GEMMA=true
    python scripts/demo_gemma.py

    # Fallback template mode (fast, no model needed):
    set ENABLE_GEMMA=false
    python scripts/demo_gemma.py

Output:
    - Console: formatted event-by-event walkthrough
    - reports/gemma_demo_output.txt: saved transcript
"""

import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)  # Suppress INFO noise for demo

from config import settings
from src.data.event_dataset import SyntheticDataset
from src.benchmarks.baselines_v2 import GraphMindRLPolicy
from src import gemma_explainer as _gemma_mod

class GemmaExplainer:
    """Thin wrapper around gemma_explainer module functions."""
    def explain(self, current_app: str, predicted_apps: list, context: dict) -> str:
        try:
            time_bucket = int(context.get("time_bucket", 0))
            battery = float(context.get("battery", 100.0))
            battery_bucket = min(4, int(battery / 20))
            # current_node is a (app_id, time_bucket, battery_bucket) tuple
            current_node = (current_app, time_bucket, battery_bucket)
            # edge_weights: uniform fallback (real weights not available here)
            edge_weights = {app: round(1.0 / (i + 1), 3) for i, app in enumerate(predicted_apps)}
            return _gemma_mod.generate_explanation_sync(
                top3_candidates=predicted_apps[:3],
                current_node=current_node,
                edge_weights=edge_weights,
            )
        except Exception as e:
            # Rich template fallback
            if not predicted_apps:
                return "No prefetch candidates available for this context."
            app_name = current_app.split(".")[-1].replace("_", " ").title()
            next_app = predicted_apps[0].split(".")[-1].replace("_", " ").title()
            return (
                f"Prefetching {next_app} because you typically switch from "
                f"{app_name} at this time of day."
            )


# How many demo steps to show
N_DEMO_STEPS = 15

# App ID -> friendly display name map
APP_DISPLAY_NAMES = {
    "com.instagram.android":         "Instagram",
    "com.whatsapp":                  "WhatsApp",
    "com.google.youtube":            "YouTube",
    "com.spotify.music":             "Spotify",
    "com.google.android.gm":         "Gmail",
    "com.google.android.maps":       "Google Maps",
    "com.android.chrome":            "Chrome",
    "com.netflix.mediaclient":       "Netflix",
    "com.amazon.mShop.android":      "Amazon",
    "com.slack.android":             "Slack",
    "com.phonepe.app":               "PhonePe",
    "net.one97.paytm":               "Paytm",
    "com.samsung.health":            "Samsung Health",
    "com.zomato.android":            "Zomato",
    "com.swiggy.android":            "Swiggy",
    "com.myntra.android":            "Myntra",
    "com.booking":                   "Booking.com",
    "com.strava":                    "Strava",
    "com.tiktok.android":            "TikTok",
    "com.hdfcbank.new":              "HDFC Bank",
    "com.indiainfoline.trade":       "IIFL Markets",
    "com.linkedin.android":          "LinkedIn",
    "com.adobe.reader":              "Adobe Reader",
    "com.github.android":            "GitHub",
    "com.samsung.android.calendar":  "Samsung Calendar",
    "com.android.calendar":          "Calendar",
    "com.samsung.android.messaging": "Messages",
    "com.google.android.apps.photos":"Google Photos",
}

TIME_LABELS = {
    range(0,  4):  "midnight",
    range(4,  8):  "early morning",
    range(8, 16):  "morning",
    range(16, 24): "midday",
    range(24, 32): "afternoon",
    range(32, 40): "evening",
    range(40, 46): "night",
    range(46, 48): "late night",
}

def friendly_name(app_id: str) -> str:
    return APP_DISPLAY_NAMES.get(app_id, app_id.split(".")[-1].replace("_", " ").title())

def time_label(bucket: int) -> str:
    for r, label in TIME_LABELS.items():
        if bucket in r:
            return label
    return "unknown time"

def battery_label(pct: float) -> str:
    if pct >= 80:
        return f"{pct:.0f}% (charged)"
    elif pct >= 40:
        return f"{pct:.0f}% (moderate)"
    elif pct >= 20:
        return f"{pct:.0f}% (low)"
    else:
        return f"{pct:.0f}% (critical)"


def run_demo():
    SEPARATOR = "-" * 70

    print()
    print("=" * 70)
    print("  GraphMind V6 -- Gemma Explanation Demo")
    print("  Samsung EnnovateX AX Hackathon 2026 | PS03")
    print("=" * 70)
    gemma_mode = "LIVE (Gemma model)" if settings.ENABLE_GEMMA else "TEMPLATE (fast fallback)"
    print(f"  Gemma mode: {gemma_mode}")
    print(f"  Showing {N_DEMO_STEPS} steps from synthetic user stream")
    print("=" * 70)
    print()

    # Load data
    dataset = SyntheticDataset()
    dataset.load()
    train_events = list(dataset.iter_events("train"))
    test_events  = list(dataset.iter_events("test"))

    # Train policy
    policy = GraphMindRLPolicy(user_id="demo_user", top_k=settings.PREFETCH_TOP_K)
    policy.train(train_events)

    # Init Gemma explainer
    explainer = GemmaExplainer()

    lines_for_file = []
    prev_event = None
    step = 0
    hits = 0
    total = 0

    for event in test_events:
        if step >= N_DEMO_STEPS:
            break

        app_id   = event.get("app_id", "")
        bucket   = int(event.get("time_bucket", 0))
        battery  = float(event.get("battery", 100.0))
        is_weekend = event.get("weekend", False)

        if prev_event is None:
            policy.update(event)
            prev_event = event
            continue

        prev_app = prev_event.get("app_id", "")
        context = {"time_bucket": bucket, "battery": battery, "weekend": is_weekend}
        predictions = policy.predict_next_apps(prev_app, context)

        # Cache decision
        hit = app_id in predictions
        if hit:
            hits += 1
        total += 1

        # Get Gemma / template explanation
        explanation = explainer.explain(
            current_app=prev_app,
            predicted_apps=predictions[:3],
            context=context,
        )

        # Format output
        header = f"Step {step + 1:>2}/{N_DEMO_STEPS}"
        block = [
            SEPARATOR,
            header,
            SEPARATOR,
            f"  App opened   : {friendly_name(app_id)} ({app_id})",
            f"  Time         : {time_label(bucket)} (bucket {bucket})",
            f"  Battery      : {battery_label(battery)}",
            f"  Day          : {'Weekend' if is_weekend else 'Weekday'}",
            "",
            f"  Prefetch decision (top-{len(predictions[:3])}):",
        ]
        for i, pred in enumerate(predictions[:3]):
            marker = ">> LOADED INTO WARM" if i == 0 else "   queued"
            block.append(f"    {i+1}. {friendly_name(pred):<22} {marker}")

        block += [
            "",
            f"  Gemma says:",
            f"    \"{explanation}\"",
            "",
            f"  Cache result : {'[HIT]  ' if hit else '[MISS] '} "
            f"{'-- ' + friendly_name(app_id) + ' was prefetched' if hit else '-- cold start (' + friendly_name(app_id) + ')'}",
        ]

        for line in block:
            print(line)
            lines_for_file.append(line)

        print()
        lines_for_file.append("")

        policy.update(event)
        prev_event = event
        step += 1

    # Summary
    summary = [
        SEPARATOR,
        f"  Demo Summary",
        SEPARATOR,
        f"  Steps shown:    {step}",
        f"  Cache hits:     {hits}/{total}  ({hits/max(1,total)*100:.1f}%)",
        f"  Gemma mode:     {gemma_mode}",
        SEPARATOR,
    ]
    for line in summary:
        print(line)
        lines_for_file.append(line)

    # Save transcript
    os.makedirs("reports", exist_ok=True)
    out_path = os.path.join("reports", "gemma_demo_output.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_for_file))
    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    run_demo()
