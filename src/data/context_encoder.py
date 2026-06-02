"""
src/data/context_encoder.py

Converts raw OS event tuples into 64-dim situation embeddings.
These become graph node features.
"""

import os
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from config import settings

logger = logging.getLogger(__name__)

# Fixed vocabulary of 30 app IDs
APP_ID_VOCAB = [
    "com.instagram.android", "com.google.youtube", "com.spotify.music",
    "com.slack.android", "com.google.android.gm", "com.linkedin.android",
    "com.google.android.maps", "com.android.calendar", "com.tiktok.android",
    "com.whatsapp", "com.netflix.mediaclient", "com.amazon.mShop.android",
    "net.one97.paytm", "com.google.android.apps.photos", "com.github.android",
    "com.samsung.health", "com.strava", "com.myntra.android",
    "com.zomato.android", "com.swiggy.android", "com.google.android.apps.docs",
    "com.adobe.reader", "com.phonepe.app", "com.hdfcbank.new",
    "com.samsung.android.messaging", "com.booking", "com.makemytrip",
    "com.indiainfoline.trade", "com.samsung.android.calendar", "unknown"
]

_APP_ID_INDEX = {app_id: idx for idx, app_id in enumerate(APP_ID_VOCAB)}


class _EncoderMLP(nn.Module):
    """Internal MLP for the ContextEncoder."""

    def __init__(self) -> None:
        """Build the 3-layer MLP architecture."""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(35, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        return self.layers(x)


class ContextEncoder:
    """
    Lightweight MLP that encodes OS event tuples into 64-dim embeddings.
    Input: (app_id_onehot[30], time_bucket[1], battery_bucket[1], headphones[1], calendar_near[1], weekend[1]) = 35 dims
    Output: 64-dim embedding vector
    Model is initialized with random weights and updated during RL training.
    """

    def __init__(self) -> None:
        """
        Define the MLP architecture using PyTorch:
        Layer 1: Linear(35, 128) + ReLU
        Layer 2: Linear(128, 64) + ReLU
        Output:  Linear(64, 64)  (no activation — raw embedding)
        Load weights from MODELS_DIR/encoder.pt if file exists.
        Set to eval mode. Use GEMMA_DEVICE for device placement.
        """
        self.device = torch.device(settings.GEMMA_DEVICE)
        torch.manual_seed(settings.RANDOM_SEED)
        self._model = _EncoderMLP().to(self.device)
        weights_path = os.path.join(settings.MODELS_DIR, "encoder.pt")
        if os.path.exists(weights_path):
            try:
                self.load_weights(weights_path)
                logger.info(f"ContextEncoder loaded weights from {weights_path}")
            except Exception as e:
                logger.warning(f"Could not load encoder weights: {e}")
        self._model.eval()

    def encode(self, event: dict) -> np.ndarray:
        """
        Convert an event dict to a 64-dim numpy embedding.
        event keys: app_id (str), time_bucket (int 0-47), battery (float),
                    headphones (bool), calendar_event_in_mins (int|None), weekend (bool)

        Encoding steps:
        1. app_id -> one-hot vector of size 30 (use APP_ID_VOCAB defined below)
        2. time_bucket -> normalize to [0,1] by dividing by 47
        3. battery -> normalize to [0,1] by dividing by 100
        4. headphones -> float 0.0 or 1.0
        5. calendar_near -> 1.0 if calendar_event_in_mins <= 30, else 0.0
        6. weekend -> float 0.0 or 1.0
        Concatenate all into tensor of shape (35,), pass through MLP, return as numpy (64,).
        """
        # 1. One-hot encode app_id
        app_id = event.get("app_id", "unknown")
        idx = _APP_ID_INDEX.get(app_id, 29)  # 29 = "unknown"
        one_hot = np.zeros(30, dtype=np.float32)
        one_hot[idx] = 1.0

        # 2-6. Scalar features
        time_norm = float(event.get("time_bucket", 0)) / 47.0
        battery_norm = float(event.get("battery", 100.0)) / 100.0
        headphones = 1.0 if event.get("headphones", False) else 0.0
        cal_mins = event.get("calendar_event_in_mins")
        calendar_near = 1.0 if (cal_mins is not None and cal_mins <= 30) else 0.0
        weekend = 1.0 if event.get("weekend", False) else 0.0

        features = np.array([time_norm, battery_norm, headphones, calendar_near, weekend],
                            dtype=np.float32)
        input_vec = np.concatenate([one_hot, features])  # shape (35,)

        with torch.no_grad():
            tensor = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
            output = self._model(tensor)
            return output.squeeze(0).cpu().numpy()

    def save_weights(self, path: str) -> None:
        """Save model state_dict to path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self._model.state_dict(), path)
        logger.debug(f"ContextEncoder weights saved to {path}")

    def load_weights(self, path: str) -> None:
        """Load model state_dict from path. Raise FileNotFoundError if missing."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Encoder weights not found: {path}")
        state = torch.load(path, map_location=self.device)
        self._model.load_state_dict(state)
        self._model.eval()
