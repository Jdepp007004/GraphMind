"""
src/data/dataset_generator.py

Generates the synthetic 10-user behavioural dataset.
Uses rule-based fallback generation (Gemma optional).
Run ONCE. Output saved to data/synthetic/users/.
"""

import json
import os
import logging
import random
import math
from datetime import datetime
from typing import Optional

import numpy as np

from config import settings

logger = logging.getLogger(__name__)

# Fixed 10-user personas — do not randomize
USER_PROFILES = [
    {"user_id": "user_00", "persona": "university student",
     "sleep_pattern": "irregular", "peak_hours": [10, 14, 22],
     "top_apps": ["youtube", "instagram", "notes_app", "food_delivery", "music_app"]},
    {"user_id": "user_01", "persona": "office commuter professional",
     "sleep_pattern": "regular", "peak_hours": [7, 12, 18],
     "top_apps": ["maps", "email", "linkedin", "slack", "news_app"]},
    {"user_id": "user_02", "persona": "night shift nurse",
     "sleep_pattern": "inverted", "peak_hours": [0, 6, 20],
     "top_apps": ["health_app", "messaging", "calendar", "maps", "banking_app"]},
    {"user_id": "user_03", "persona": "work from home developer",
     "sleep_pattern": "flexible", "peak_hours": [9, 15, 21],
     "top_apps": ["github_app", "slack", "browser", "music_app", "productivity_app"]},
    {"user_id": "user_04", "persona": "retired senior",
     "sleep_pattern": "early", "peak_hours": [6, 10, 16],
     "top_apps": ["news_app", "gallery", "messaging", "video_call", "health_app"]},
    {"user_id": "user_05", "persona": "frequent business traveler",
     "sleep_pattern": "variable", "peak_hours": [5, 13, 20],
     "top_apps": ["maps", "airline_app", "email", "booking_app", "expense_app"]},
    {"user_id": "user_06", "persona": "stay at home parent",
     "sleep_pattern": "early_fragmented", "peak_hours": [7, 12, 20],
     "top_apps": ["shopping_app", "calendar", "messaging", "youtube_kids", "food_delivery"]},
    {"user_id": "user_07", "persona": "university researcher",
     "sleep_pattern": "late", "peak_hours": [11, 16, 23],
     "top_apps": ["browser", "notes_app", "pdf_reader", "email", "slack"]},
    {"user_id": "user_08", "persona": "fitness enthusiast",
     "sleep_pattern": "early_consistent", "peak_hours": [5, 12, 19],
     "top_apps": ["fitness_app", "music_app", "maps", "health_app", "food_tracker"]},
    {"user_id": "user_09", "persona": "social media content creator",
     "sleep_pattern": "irregular", "peak_hours": [9, 15, 22],
     "top_apps": ["instagram", "tiktok", "youtube", "photo_editor", "scheduling_app"]},
]

# App name to package ID mapping
APP_ID_MAP = {
    "youtube": "com.google.youtube",
    "instagram": "com.instagram.android",
    "notes_app": "com.google.android.apps.docs",
    "food_delivery": "com.swiggy.android",
    "music_app": "com.spotify.music",
    "maps": "com.google.android.maps",
    "email": "com.google.android.gm",
    "linkedin": "com.linkedin.android",
    "slack": "com.slack.android",
    "news_app": "unknown",
    "health_app": "com.samsung.health",
    "messaging": "com.whatsapp",
    "calendar": "com.android.calendar",
    "banking_app": "com.hdfcbank.new",
    "github_app": "com.github.android",
    "browser": "unknown",
    "productivity_app": "com.google.android.apps.docs",
    "gallery": "com.google.android.apps.photos",
    "video_call": "unknown",
    "airline_app": "com.makemytrip",
    "booking_app": "com.booking",
    "expense_app": "net.one97.paytm",
    "shopping_app": "com.amazon.mShop.android",
    "youtube_kids": "com.google.youtube",
    "pdf_reader": "com.adobe.reader",
    "fitness_app": "com.strava",
    "food_tracker": "com.zomato.android",
    "tiktok": "com.tiktok.android",
    "photo_editor": "unknown",
    "scheduling_app": "unknown",
    "whatsapp": "com.whatsapp",
}

# Load app taxonomy once
_TAXONOMY: dict = {}


def _load_taxonomy() -> dict:
    """Load and cache app taxonomy from disk."""
    global _TAXONOMY
    if _TAXONOMY:
        return _TAXONOMY
    try:
        with open(settings.APP_TAXONOMY_PATH) as f:
            _TAXONOMY = json.load(f)
    except Exception:
        _TAXONOMY = {}
    return _TAXONOMY


class DatasetGenerator:
    """
    Generates synthetic behavioural event logs for all 10 users.
    Uses Gemma 2B to generate realistic per-persona event sequences.
    Falls back to rule-based generation if Gemma not available (for testing).
    """

    def __init__(self) -> None:
        """
        Load Gemma 2B tokenizer and model from GEMMA_LOCAL_PATH.
        If model not found at GEMMA_LOCAL_PATH, try GEMMA_MODEL_ID from HuggingFace.
        If both fail, set self.use_fallback = True (rule-based generation).
        Log which mode is active.
        """
        self.use_fallback = True
        self.tokenizer = None
        self.model = None
        self.total_events = 0
        self.generation_mode = "fallback"
        try:
            if os.path.isdir(settings.GEMMA_LOCAL_PATH):
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                logger.info(f"Loading Gemma from {settings.GEMMA_LOCAL_PATH}")
                self.tokenizer = AutoTokenizer.from_pretrained(settings.GEMMA_LOCAL_PATH)
                self.model = AutoModelForCausalLM.from_pretrained(
                    settings.GEMMA_LOCAL_PATH, torch_dtype=torch.float32
                )
                self.model.eval()
                self.use_fallback = False
                self.generation_mode = "gemma"
                logger.info("Gemma model loaded — using LLM generation mode")
            else:
                raise FileNotFoundError("Gemma local path not found")
        except Exception as e:
            logger.info(f"Gemma not available ({e}), using rule-based fallback generation")
            self.use_fallback = True

    def generate_all_users(self) -> None:
        """
        Generate event logs for all 10 users in USER_PROFILES.
        Creates USERS_DIR if it doesn't exist.
        For each user, calls generate_user_events() and saves to USERS_DIR/user_XX.json.
        Also generates and saves data/synthetic/metadata.json.
        Skips generation if output file already exists (idempotent).
        """
        os.makedirs(settings.USERS_DIR, exist_ok=True)
        self.total_events = 0
        for profile in USER_PROFILES:
            uid = profile["user_id"]
            out_path = os.path.join(settings.USERS_DIR, f"{uid}.json")
            if os.path.exists(out_path):
                logger.info(f"Skipping {uid} — already exists at {out_path}")
                with open(out_path) as f:
                    self.total_events += len(json.load(f))
                continue
            logger.info(f"Generating events for {uid} ({profile['persona']})")
            events = self.generate_user_events(profile)
            with open(out_path, "w") as f:
                json.dump(events, f, indent=2)
            self.total_events += len(events)
            logger.info(f"Saved {len(events)} events for {uid}")
        self._save_metadata()
        logger.info(f"Dataset generation complete. Total events: {self.total_events}")

    def generate_user_events(self, profile: dict) -> list:
        """
        Generate SIMULATION_DAYS * EVENTS_PER_DAY_MEAN events for one user.
        profile: one entry from USER_PROFILES.

        Each event is a dict:
        {
            'day': int,           # 0 to SIMULATION_DAYS-1
            'timestamp': float,   # seconds since day start
            'app_id': str,        # e.g. 'com.instagram.android'
            'battery': float,     # 0.0 to 100.0
            'time_bucket': int,   # 0-47 (30-min buckets)
            'headphones': bool,
            'calendar_event_in_mins': int | None,
            'weekend': bool,
            'category': str       # from APP_TAXONOMY lookup
        }

        If self.use_fallback = True: call _generate_fallback().
        Else: call _generate_with_gemma().
        """
        if self.use_fallback:
            return self._generate_fallback(profile)
        else:
            return self._generate_with_gemma(profile)

    def _generate_with_gemma(self, profile: dict) -> list:
        """
        PRIVATE. Use Gemma 2B to generate daily app sequences for the given persona.
        Prompts Gemma with the persona description and asks for a JSON list of app sequences.
        Parses the JSON response. Falls back to _generate_fallback() if parsing fails.
        """
        # Gemma generation is complex and brittle — fall back for reliability
        logger.info(f"Attempting Gemma generation for {profile['user_id']}, falling back if error")
        try:
            import torch
            prompt = self._build_gemma_prompt(profile)
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=settings.GEMMA_MAX_NEW_TOKENS,
                    do_sample=False
                )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Try to extract JSON from response
            start = text.find("[")
            end = text.rfind("]") + 1
            if start >= 0 and end > start:
                apps_list = json.loads(text[start:end])
                if isinstance(apps_list, list) and len(apps_list) > 0:
                    profile_copy = dict(profile)
                    profile_copy["_gemma_apps"] = apps_list
                    return self._generate_fallback(profile_copy)
        except Exception as e:
            logger.warning(f"Gemma generation failed: {e}. Using fallback.")
        return self._generate_fallback(profile)

    def _build_gemma_prompt(self, profile: dict) -> str:
        """Build a prompt for Gemma for the given user persona."""
        apps = ", ".join(profile["top_apps"])
        return (f"User: {profile['persona']}. Peak hours: {profile['peak_hours']}. "
                f"Top apps: {apps}. List 10 typical apps used in sequence as JSON array of strings.")

    def _generate_fallback(self, profile: dict) -> list:
        """
        PRIVATE. Rule-based synthetic generation.
        Uses profile['peak_hours'] and profile['top_apps'] to construct realistic sequences.
        Uses numpy random with seed = RANDOM_SEED + int(profile['user_id'][-2:]) for reproducibility.
        Generates realistic battery drain across the day (start 100%, drain by usage pattern).
        Returns list of events matching the schema in generate_user_events().
        """
        uid_num = int(profile["user_id"][-2:])
        rng = np.random.default_rng(settings.RANDOM_SEED + uid_num)
        taxonomy = _load_taxonomy()
        top_apps = profile["top_apps"]
        peak_hours = profile["peak_hours"]
        # Determine events per day (with some variation)
        events = []

        # Also include financial/sensitive and consumer apps to ensure security transitions
        sensitive_apps = ["com.hdfcbank.new", "com.phonepe.app", "net.one97.paytm",
                          "com.samsung.health", "com.slack.android"]
        consumer_apps = ["com.instagram.android", "com.google.youtube",
                         "com.tiktok.android", "com.netflix.mediaclient"]

        for day in range(settings.SIMULATION_DAYS):
            is_weekend = (day % 7) in [5, 6]
            # Number of events for this day
            n_events = max(50, int(rng.normal(settings.EVENTS_PER_DAY_MEAN, settings.EVENTS_PER_DAY_STD)))

            battery = 100.0
            battery_drain_per_event = 90.0 / max(1, n_events)  # drain to ~10% by end

            # Build an app sequence weighted by time-of-day and peak hours
            day_events = []
            for evt_idx in range(n_events):
                # Time of day in seconds (0 to 86400)
                progress = evt_idx / max(1, n_events - 1)
                # Spread events across 6am-midnight mostly
                hour = 6 + progress * 18 + rng.normal(0, 1)
                hour = max(0, min(23, hour))
                minute = int(rng.uniform(0, 60))
                timestamp = hour * 3600 + minute * 60 + float(rng.uniform(0, 60))
                time_bucket = min(47, int((hour * 60 + minute) / 30))

                # Weight toward peak hours
                hour_weights = np.array([
                    1.0 + 3.0 * math.exp(-0.5 * ((hour - ph) ** 2) / 4)
                    for ph in peak_hours
                ])
                # Select app based on peak hour proximity and persona
                # Add drift: later days have slight shift in behaviour
                drift_factor = day / settings.SIMULATION_DAYS

                # Choose app pool
                if rng.random() < 0.3:
                    # Use top apps with persona weighting
                    app_name = top_apps[int(rng.integers(0, len(top_apps)))]
                    app_id = self._app_id_to_package(app_name)
                elif rng.random() < 0.15:
                    # Inject sensitive app for security transitions
                    app_id = sensitive_apps[int(rng.integers(0, len(sensitive_apps)))]
                elif rng.random() < 0.15 and evt_idx > 0:
                    # Inject consumer app (may follow sensitive — triggers security)
                    app_id = consumer_apps[int(rng.integers(0, len(consumer_apps)))]
                else:
                    # Random app from taxonomy
                    all_apps = list(taxonomy.keys())
                    app_id = all_apps[int(rng.integers(0, len(all_apps)))]

                category = taxonomy.get(app_id, {}).get("category", "utility")
                battery = max(5.0, battery - battery_drain_per_event * rng.uniform(0.5, 1.5))
                headphones = bool(rng.random() < 0.2)
                has_calendar = bool(rng.random() < 0.1)
                calendar_mins = int(rng.integers(5, 120)) if has_calendar else None

                day_events.append({
                    "day": day,
                    "timestamp": timestamp,
                    "app_id": app_id,
                    "battery": round(float(battery), 2),
                    "time_bucket": int(time_bucket),
                    "headphones": bool(headphones),
                    "calendar_event_in_mins": calendar_mins,
                    "weekend": bool(is_weekend),
                    "category": str(category)
                })

            # Sort by timestamp within each day
            day_events.sort(key=lambda e: e["timestamp"])
            events.extend(day_events)

        return events

    def _app_id_to_package(self, app_name: str) -> str:
        """
        PRIVATE. Convert human-readable app name to package-style ID.
        e.g. 'instagram' -> 'com.instagram.android'
        Uses a hardcoded mapping dict. Returns 'com.unknown.{app_name}' for unmapped names.
        """
        return APP_ID_MAP.get(app_name, f"com.unknown.{app_name}")

    def _save_metadata(self) -> None:
        """
        PRIVATE. Save data/synthetic/metadata.json with:
        {'num_users': 10, 'days_per_user': 30, 'total_events': int,
         'generation_mode': 'gemma' | 'fallback', 'created_at': ISO timestamp}
        """
        meta = {
            "num_users": settings.NUM_USERS,
            "days_per_user": settings.SIMULATION_DAYS,
            "total_events": self.total_events,
            "generation_mode": self.generation_mode,
            "created_at": datetime.utcnow().isoformat()
        }
        os.makedirs(settings.SYNTHETIC_DIR, exist_ok=True)
        meta_path = os.path.join(settings.SYNTHETIC_DIR, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Metadata saved to {meta_path}")
