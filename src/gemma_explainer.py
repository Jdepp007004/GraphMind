"""
src/gemma_explainer.py

Gemma Explanation Layer — GraphMind V5
======================================

Generates one-sentence natural-language explanations for each prefetch decision
using the Gemma language model (google/gemma-2b or 3b-it as available).

Architecture role: Tool Use #2 in the GraphMind agentic pipeline.
  Tool 1: BehaviouralGraph.query(node)          → transition distribution
  Tool 2: Gemma.generate_explanation(candidates) → natural language string

CRITICAL DESIGN INVARIANT:
  This module fires AFTER the prefetch decision is already made.
  It has zero effect on F1, cache_hit_rate, or any benchmark metric.
  All benchmarks run identically regardless of ENABLE_GEMMA.
  This is enforced by the evaluator_v2.py ENABLE_GEMMA guard.

Usage:
    from src.gemma_explainer import generate_explanation

    explanation = await generate_explanation(
        top3_candidates=["com.spotify.music", "com.whatsapp", "com.instagram.android"],
        current_node=("com.google.youtube", 6, 4),  # (app_id, time_bucket, battery_bucket)
        edge_weights={"com.spotify.music": 0.72, "com.whatsapp": 0.58},
    )
    # → "Preloading Spotify because you typically switch from YouTube after 8pm on weekdays."

Configuration:
    ENABLE_GEMMA = True  in config/settings.py   (set by ENABLE_GEMMA env var)
    GEMMA_MODEL_ID       = "google/gemma-2b"
    GEMMA_MAX_NEW_TOKENS = 128
    GEMMA_DEVICE         = "cpu"  (or "cuda" if GPU available)

Fallback:
    If Gemma is unavailable (model not downloaded, OOM, or ENABLE_GEMMA=False),
    returns a deterministic template string derived from the edge weights directly.
    The template fallback is always safe to call and never raises.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)

# ── Time-bucket → human label ──────────────────────────────────────────────────
# UbiqLog uses 30-minute buckets: 0=midnight, 47=11:30pm
_BUCKET_TO_LABEL: Dict[int, str] = {
    **{b: "late at night" for b in range(0, 6)},      # 0–5  → midnight–2:30am
    **{b: "in the early morning" for b in range(6, 14)},   # 6–13  → 3am–6:30am
    **{b: "in the morning" for b in range(14, 22)},   # 14–21 → 7am–10:30am
    **{b: "around midday" for b in range(22, 30)},    # 22–29 → 11am–2:30pm
    **{b: "in the afternoon" for b in range(30, 38)}, # 30–37 → 3pm–6:30pm
    **{b: "in the evening" for b in range(38, 44)},   # 38–43 → 7pm–9:30pm
    **{b: "at night" for b in range(44, 48)},         # 44–47 → 10pm–11:30pm
}

# ── Battery-bucket → human label ──────────────────────────────────────────────
# Buckets: 0 = 0–20%, 1 = 20–40%, 2 = 40–60%, 3 = 60–80%, 4 = 80–100%
_BATTERY_TO_LABEL: Dict[int, str] = {
    0: "with low battery",
    1: "with moderate battery",
    2: "with about half charge",
    3: "with plenty of charge",
    4: "while fully charged",
}

# ── App-id → friendly display name (coverage for top UbiqLog apps) ─────────────
_APP_DISPLAY: Dict[str, str] = {
    "com.instagram.android":    "Instagram",
    "com.whatsapp":             "WhatsApp",
    "com.google.youtube":       "YouTube",
    "com.spotify.music":        "Spotify",
    "com.google.android.gm":    "Gmail",
    "com.google.android.maps":  "Google Maps",
    "com.android.chrome":       "Chrome",
    "com.netflix.mediaclient":  "Netflix",
    "com.amazon.mShop.android": "Amazon",
    "com.slack.android":        "Slack",
    "com.phonepe.app":          "PhonePe",
    "net.one97.paytm":          "Paytm",
    "com.samsung.health":       "Samsung Health",
    "com.google.android.dialer":"Phone",
    "com.samsung.android.messaging": "Messages",
    "com.google.android.apps.photos": "Google Photos",
}


def _app_name(app_id: str) -> str:
    """Return a human-readable app name from its package ID."""
    if app_id in _APP_DISPLAY:
        return _APP_DISPLAY[app_id]
    # Heuristic: use the last component of the package name, title-cased
    parts = app_id.split(".")
    if parts:
        return parts[-1].replace("_", " ").title()
    return app_id


def _time_label(time_bucket: int) -> str:
    """Return a human-readable time-of-day label for a 30-min bucket index."""
    return _BUCKET_TO_LABEL.get(int(time_bucket), "during the day")


def _battery_label(battery_bucket: int) -> str:
    """Return a human-readable battery-level label."""
    return _BATTERY_TO_LABEL.get(int(battery_bucket), "")


def _build_template_explanation(
    top3_candidates: List[str],
    current_node: Tuple[str, int, int],
    edge_weights: Dict[str, float],
) -> str:
    """
    Generate a deterministic template explanation without Gemma.

    This is the guaranteed fallback — it always runs and never raises.
    Used when: ENABLE_GEMMA=False, model unavailable, or OOM.

    Args:
        top3_candidates: Up to 3 candidate app IDs ranked by confidence score.
        current_node:    (app_id, time_bucket, battery_bucket) tuple.
        edge_weights:    Dict of app_id → confidence score for top candidates.

    Returns:
        str: One-sentence natural-language explanation.
    """
    if not top3_candidates:
        return "Optimising memory allocation based on your usage history."

    top_app = top3_candidates[0]
    current_app_id, time_bucket, battery_bucket = current_node
    time_lbl = _time_label(time_bucket)
    battery_lbl = _battery_label(battery_bucket)
    weight = edge_weights.get(top_app, 0.0)

    current_name = _app_name(current_app_id)
    top_name = _app_name(top_app)

    # Prefer transition-based phrasing when weight is high enough
    if weight >= 0.60:
        return (
            f"Preloading {top_name} because you almost always switch from "
            f"{current_name} {time_lbl}."
        )
    elif weight >= 0.35:
        return (
            f"Preloading {top_name} because you frequently open it after "
            f"{current_name} {time_lbl}{' ' + battery_lbl if battery_lbl else ''}."
        )
    else:
        # Low-confidence: frequency-based phrasing
        if len(top3_candidates) >= 2:
            second_name = _app_name(top3_candidates[1])
            return (
                f"Preloading {top_name} and {second_name} based on your "
                f"most-used apps {time_lbl}."
            )
        return (
            f"Preloading {top_name} based on your usage patterns {time_lbl}."
        )


def _build_gemma_prompt(
    top3_candidates: List[str],
    current_node: Tuple[str, int, int],
    edge_weights: Dict[str, float],
) -> str:
    """
    Construct the Gemma prompt for explanation generation.

    The prompt instructs Gemma to produce exactly one sentence in
    user-facing natural language — not technical jargon.

    Args:
        top3_candidates: Ranked candidate app IDs.
        current_node:    (app_id, time_bucket, battery_bucket) tuple.
        edge_weights:    Confidence scores for top candidates.

    Returns:
        str: The full prompt string to send to Gemma.
    """
    current_app_id, time_bucket, battery_bucket = current_node
    current_name = _app_name(current_app_id)
    time_lbl = _time_label(time_bucket)
    battery_lbl = _battery_label(battery_bucket)

    candidates_str = ", ".join(
        f"{_app_name(a)} (confidence: {edge_weights.get(a, 0.0):.2f})"
        for a in top3_candidates[:3]
    )

    return (
        f"You are a smart phone assistant explaining a memory prefetch decision "
        f"to a user in one short, friendly sentence.\n\n"
        f"The user just opened {current_name}. "
        f"The time is {time_lbl}. Battery is {battery_lbl}.\n"
        f"The system will preload these apps next: {candidates_str}.\n\n"
        f"Write exactly one sentence explaining why the top app will be preloaded. "
        f"Use simple, friendly language. Do not mention 'confidence', 'algorithm', "
        f"'Markov', 'cache', or 'prefetch'.\n\n"
        f"Explanation:"
    )


# ── Model singleton ────────────────────────────────────────────────────────────
# Loaded once on first call; subsequent calls reuse the loaded model.
_gemma_model = None
_gemma_tokenizer = None
_gemma_load_attempted = False


def _load_gemma_model():
    """
    Load Gemma model and tokenizer from local path or HuggingFace Hub.

    Sets module-level _gemma_model and _gemma_tokenizer.
    Sets _gemma_load_attempted = True regardless of outcome.
    On failure: logs the error and leaves model as None (triggers fallback).
    """
    global _gemma_model, _gemma_tokenizer, _gemma_load_attempted
    _gemma_load_attempted = True

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        model_id = settings.GEMMA_MODEL_ID
        local_path = settings.GEMMA_LOCAL_PATH
        device = settings.GEMMA_DEVICE

        # Prefer local path if it exists and has model files
        import os
        source = (
            local_path
            if os.path.isdir(local_path) and os.listdir(local_path)
            else model_id
        )
        logger.info(f"Loading Gemma from: {source} → device={device}")

        _gemma_tokenizer = AutoTokenizer.from_pretrained(source)
        _gemma_model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=torch.float32 if device == "cpu" else torch.float16,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        _gemma_model.eval()
        logger.info("Gemma model loaded successfully.")

    except Exception as exc:
        logger.warning(
            f"Gemma model failed to load: {exc}. "
            "Explanation layer will use template fallback."
        )
        _gemma_model = None
        _gemma_tokenizer = None


def _run_gemma_inference(prompt: str) -> str:
    """
    Run one inference pass through Gemma and extract the explanation sentence.

    Args:
        prompt: The full prompt string from _build_gemma_prompt().

    Returns:
        str: The generated explanation sentence (stripped of the prompt prefix).

    Raises:
        RuntimeError: If the model is None (caller should fall back to template).
    """
    if _gemma_model is None or _gemma_tokenizer is None:
        raise RuntimeError("Gemma model not loaded")

    import torch

    device = settings.GEMMA_DEVICE
    inputs = _gemma_tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = _gemma_model.generate(
            **inputs,
            max_new_tokens=settings.GEMMA_MAX_NEW_TOKENS,
            do_sample=False,          # greedy decoding for determinism
            temperature=1.0,
            pad_token_id=_gemma_tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = _gemma_tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Extract first sentence only
    match = re.search(r"[^.!?]*[.!?]", text)
    if match:
        return match.group(0).strip()
    # Fallback: return the first 150 characters
    return text[:150].strip()


async def generate_explanation(
    top3_candidates: List[str],
    current_node: Tuple[str, int, int],
    edge_weights: Dict[str, float],
) -> str:
    """
    Generate a one-sentence natural-language explanation for a prefetch decision.

    This is Tool Use #2 in the GraphMind agentic pipeline. It fires AFTER the
    prefetch decision is already made. It has no effect on any benchmark metric.

    Args:
        top3_candidates: Up to 3 candidate app IDs ranked by confidence score.
                         e.g. ["com.spotify.music", "com.whatsapp", "com.instagram.android"]
        current_node:    The current node identity as a (app_id, time_bucket, battery_bucket)
                         tuple. time_bucket is a 30-minute UbiqLog bucket (0–47).
                         battery_bucket is a 20%-increment bucket (0–4).
                         e.g. ("com.google.youtube", 38, 3)
        edge_weights:    Dict mapping app_id → confidence score for the top candidates.
                         e.g. {"com.spotify.music": 0.72, "com.whatsapp": 0.58}

    Returns:
        str: One-sentence natural-language explanation suitable for display to a user.
             e.g. "Preloading Spotify because you typically switch from YouTube after 8pm."

    Fallback behaviour:
        If ENABLE_GEMMA is False, or if the Gemma model is unavailable, or if inference
        raises any exception, a deterministic template string is returned instead.
        The fallback never raises and is always safe to call.

    Example:
        explanation = await generate_explanation(
            top3_candidates=["com.spotify.music", "com.whatsapp"],
            current_node=("com.google.youtube", 38, 3),
            edge_weights={"com.spotify.music": 0.72, "com.whatsapp": 0.44},
        )
        # → "Preloading Spotify because you typically switch from YouTube in the evening."
    """
    # Fast path: ENABLE_GEMMA=False → always return template without loading model
    if not settings.ENABLE_GEMMA:
        return _build_template_explanation(top3_candidates, current_node, edge_weights)

    # Lazy-load the model on first call
    global _gemma_load_attempted
    if not _gemma_load_attempted:
        # Run model loading in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _load_gemma_model)

    # If model failed to load, fall back to template
    if _gemma_model is None:
        logger.debug("Gemma unavailable — using template explanation.")
        return _build_template_explanation(top3_candidates, current_node, edge_weights)

    # Generate the prompt
    prompt = _build_gemma_prompt(top3_candidates, current_node, edge_weights)

    try:
        # Run inference in thread pool to keep async loop non-blocking
        loop = asyncio.get_event_loop()
        explanation = await loop.run_in_executor(
            None, _run_gemma_inference, prompt
        )
        logger.debug(f"Gemma explanation: {explanation}")
        return explanation

    except Exception as exc:
        logger.warning(f"Gemma inference failed: {exc}. Falling back to template.")
        return _build_template_explanation(top3_candidates, current_node, edge_weights)


def generate_explanation_sync(
    top3_candidates: List[str],
    current_node: Tuple[str, int, int],
    edge_weights: Dict[str, float],
) -> str:
    """
    Synchronous wrapper for generate_explanation().

    For use in non-async contexts (e.g., dashboard data generation scripts).
    Creates a new event loop if one is not already running.

    Args:
        Same as generate_explanation().

    Returns:
        str: One-sentence explanation.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already inside an async context — schedule as a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    generate_explanation(top3_candidates, current_node, edge_weights),
                )
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(
                generate_explanation(top3_candidates, current_node, edge_weights)
            )
    except Exception as exc:
        logger.warning(f"generate_explanation_sync failed: {exc}. Using template.")
        return _build_template_explanation(top3_candidates, current_node, edge_weights)
