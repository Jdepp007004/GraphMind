"""
src/benchmarks/baselines_extra.py

GraphMind V6 -- Additional baseline policies for comprehensive comparison.

Baselines:
    ARIMAPolicy    -- ARIMA(1,1,1) per-user per-app time-series prediction.
    LSTMPolicy     -- 2-layer LSTM sequence predictor (PyTorch).
    ProphetPolicy  -- Facebook Prophet per-app usage forecasting.

Model persistence:
    Trained models are saved to models/saved/ on first run and loaded
    on every subsequent run -- no retraining needed when cloning the repo.

    Cache files:
        models/saved/arima_{tag}.pkl
        models/saved/lstm_{tag}.pt  +  models/saved/lstm_{tag}_meta.pkl
        models/saved/prophet_{tag}.pkl
        models/saved/v6_reranker_{tag}.pt  (handled in v6_pipeline.py)

    where tag = "ubiqlog" | "synthetic" | "custom_{N}".

Speed optimisations vs. original:
    LSTM    -- 5 epochs (was 15), max 50K training events, hidden=32 (was 64)
    Prophet -- top 300 apps only (rest use mean-count fallback)
    ARIMA   -- unchanged (already fast); just adds caching
"""

import logging
import os
import pickle
import warnings
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import numpy as np

from config import settings

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Module-level cache control
# ---------------------------------------------------------------------------

#: Absolute path to the directory where trained models are persisted.
_SAVED_MODELS_DIR: str = os.path.join(settings.PROJECT_ROOT, "models", "saved")

#: When True every policy retrains from scratch, overwriting any saved cache.
_FORCE_RETRAIN: bool = False


def set_force_retrain(val: bool) -> None:
    """
    Call before running the evaluator to control caching behaviour.

    Args:
        val: True  -> retrain all models from scratch, overwrite cache.
             False -> load cached models when available (default).
    """
    global _FORCE_RETRAIN
    _FORCE_RETRAIN = val


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dataset_tag(n_events: int) -> str:
    """Return a human-readable tag based on training-set size."""
    if n_events > 400_000:
        return "ubiqlog"
    if n_events < 50_000:
        return "synthetic"
    return f"custom_{n_events}"


def _pkl_path(model_name: str, tag: str) -> str:
    os.makedirs(_SAVED_MODELS_DIR, exist_ok=True)
    return os.path.join(_SAVED_MODELS_DIR, f"{model_name}_{tag}.pkl")


def _pt_path(model_name: str, tag: str, suffix: str = "") -> str:
    os.makedirs(_SAVED_MODELS_DIR, exist_ok=True)
    return os.path.join(_SAVED_MODELS_DIR, f"{model_name}_{tag}{suffix}.pt")


def _load_pkl(path: str) -> Optional[dict]:
    """
    Attempt to load a pickle cache.
    Returns None if missing, unreadable, or _FORCE_RETRAIN is set.
    """
    if _FORCE_RETRAIN:
        return None
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        logger.info(f"  [cache] Loaded {os.path.basename(path)}")
        return data
    except Exception as exc:
        logger.warning(f"  [cache] Load failed ({exc}), will retrain.")
        return None


def _save_pkl(path: str, data: dict) -> None:
    """Save a dict to a pickle cache file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(data, fh, protocol=4)
        logger.info(f"  [cache] Saved  {os.path.basename(path)}")
    except Exception as exc:
        logger.warning(f"  [cache] Save failed: {exc}")


def _try_tqdm(iterable, **kwargs):
    """Wrap iterable with tqdm if available, otherwise return as-is."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


# ---------------------------------------------------------------------------
# ARIMA Baseline
# ---------------------------------------------------------------------------

class ARIMAPolicy:
    """
    ARIMA(1,1,1) time-series baseline policy.

    Per app: fits ARIMA on 48-bin half-hourly usage counts.
    At prediction time: ranks apps by their 1-step-ahead forecast.
    Falls back to frequency ranking when statsmodels is unavailable.

    Caching: forecasts are persisted to models/saved/arima_{tag}.pkl so
    subsequent runs load in milliseconds.
    """

    def __init__(self, top_k: int = 8) -> None:
        self.top_k = top_k
        self._forecasts: Dict[str, float] = {}
        self._freq: Counter = Counter()
        self._arima_available = False

        try:
            from statsmodels.tsa.arima.model import ARIMA as _A
            self._ARIMA = _A
            self._arima_available = True
        except ImportError:
            logger.warning("ARIMAPolicy: statsmodels not installed -- frequency fallback active.")

    def get_name(self) -> str:
        return "ARIMA"

    def reset(self) -> None:
        self._forecasts.clear()
        self._freq.clear()

    def train(self, events: list) -> None:
        """
        Build per-app ARIMA forecasts from training events.
        Loads from cache if available; saves to cache after training.
        """
        tag = _dataset_tag(len(events))
        cache_path = _pkl_path("arima", tag)

        # -- Try cache first --------------------------------------------------
        cached = _load_pkl(cache_path)
        if cached is not None:
            self._forecasts = cached["forecasts"]
            self._freq = Counter(cached["freq"])
            logger.info(f"ARIMAPolicy: {len(self._forecasts)} forecasts loaded from cache.")
            return

        # -- Train from scratch -----------------------------------------------
        self.reset()
        for event in events:
            self._freq[event.get("app_id", "")] += 1

        if not self._arima_available:
            logger.info("ARIMAPolicy: ARIMA unavailable, storing frequency fallback in cache.")
            _save_pkl(cache_path, {"forecasts": {}, "freq": dict(self._freq)})
            return

        N_BINS = 48
        app_hourly: Dict[str, list] = defaultdict(list)
        for event in events:
            app_id = event.get("app_id", "")
            tb = int(event.get("time_bucket", 0))
            app_hourly[app_id].append(tb)

        app_series: Dict[str, np.ndarray] = {}
        for app_id, hours in app_hourly.items():
            series = np.zeros(N_BINS, dtype=float)
            for h in hours:
                series[h % N_BINS] += 1
            app_series[app_id] = series

        fitted = 0
        items = list(app_series.items())
        for app_id, series in _try_tqdm(items, desc="ARIMA: fitting", unit="app", leave=False):
            try:
                model = self._ARIMA(series, order=(1, 1, 1))
                result = model.fit(disp=False)
                forecast = result.forecast(steps=1)[0]
                self._forecasts[app_id] = max(0.0, float(forecast))
                fitted += 1
            except Exception:
                self._forecasts[app_id] = float(np.mean(series))

        logger.info(f"ARIMAPolicy: fitted {fitted}/{len(app_series)} app models.")
        _save_pkl(cache_path, {"forecasts": dict(self._forecasts), "freq": dict(self._freq)})

    def predict_next_apps(self, current_app: str, context: dict) -> List[str]:
        if self._forecasts:
            ranked = sorted(self._forecasts.items(), key=lambda x: x[1], reverse=True)
            return [app for app, _ in ranked[: self.top_k]]
        return [app for app, _ in self._freq.most_common(self.top_k)]

    def update(self, event: dict) -> None:
        self._freq[event.get("app_id", "")] += 1


# ---------------------------------------------------------------------------
# LSTM Baseline
# ---------------------------------------------------------------------------

class LSTMPolicy:
    """
    2-layer LSTM sequence predictor (PyTorch).

    Architecture:
        Embedding(vocab, 32) -> LSTM(32, hidden=32, layers=2) -> Linear(32, vocab)

    Speed optimisations:
        - max 50,000 training events (subsampled evenly when larger)
        - 5 training epochs (was 15)
        - BPTT chunk size 256 (was 128)
        - hidden_dim = 32 (was 64)

    Caching:
        Model state dict:  models/saved/lstm_{tag}.pt
        Vocabulary meta:   models/saved/lstm_{tag}_meta.pkl
    """

    _MAX_TRAIN_EVENTS: int = 50_000

    def __init__(
        self,
        top_k: int = 8,
        hidden_dim: int = 32,
        n_layers: int = 2,
        n_epochs: int = 5,
        lr: float = 5e-3,
    ) -> None:
        self.top_k = top_k
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        self.lr = lr

        self._model = None
        self._app_vocab: List[str] = []
        self._app_to_idx: dict = {}
        self._freq: Counter = Counter()
        self._history: List[int] = []
        self._torch_available = False

        try:
            import torch  # noqa: F401
            self._torch_available = True
        except ImportError:
            logger.warning("LSTMPolicy: PyTorch not installed -- frequency fallback active.")

    def get_name(self) -> str:
        return "LSTM"

    def reset(self) -> None:
        self._model = None
        self._freq.clear()
        self._history.clear()

    def _build_model(self, vocab_size: int):
        """Construct the LSTM nn.Module."""
        import torch.nn as nn

        class _LSTMModel(nn.Module):
            def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, n_layers: int):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.lstm = nn.LSTM(
                    embed_dim, hidden_dim, n_layers,
                    batch_first=True,
                    dropout=0.2 if n_layers > 1 else 0.0,
                )
                self.head = nn.Linear(hidden_dim, vocab_size)

            def forward(self, x, hidden=None):
                emb = self.embed(x)
                out, hidden = self.lstm(emb, hidden)
                return self.head(out), hidden

        return _LSTMModel(vocab_size, embed_dim=32, hidden_dim=self.hidden_dim, n_layers=self.n_layers)

    def train(self, events: list) -> None:
        """
        Train LSTM on event sequence.
        Loads from cache if available; saves to cache after training.
        """
        tag = _dataset_tag(len(events))
        pt_path   = _pt_path("lstm", tag)
        meta_path = _pkl_path("lstm", tag)

        # Always rebuild frequency counter (used for fallback predictions)
        self.reset()
        for event in events:
            self._freq[event.get("app_id", "")] += 1

        # -- Try cache first --------------------------------------------------
        if not _FORCE_RETRAIN and os.path.exists(pt_path) and os.path.exists(meta_path):
            meta = _load_pkl(meta_path)
            if meta is not None:
                try:
                    import torch
                    self._app_vocab  = meta["app_vocab"]
                    self._app_to_idx = meta["app_to_idx"]
                    vocab_size = len(self._app_vocab) + 1
                    self._model = self._build_model(vocab_size)
                    self._model.load_state_dict(
                        torch.load(pt_path, map_location="cpu", weights_only=True)
                    )
                    self._model.eval()
                    logger.info(f"LSTMPolicy: model loaded from cache (vocab={vocab_size}).")
                    return
                except Exception as exc:
                    logger.warning(f"LSTMPolicy: cache load failed ({exc}), retraining.")

        # -- Train from scratch -----------------------------------------------
        if not self._torch_available or len(events) < 20:
            return

        import torch
        import torch.nn as nn

        # Subsample training events evenly to keep training fast
        train_events = events
        if len(events) > self._MAX_TRAIN_EVENTS:
            step = max(1, len(events) // self._MAX_TRAIN_EVENTS)
            train_events = events[::step][: self._MAX_TRAIN_EVENTS]
            logger.info(f"LSTMPolicy: subsampled {len(events):,} -> {len(train_events):,} events.")

        all_apps = sorted({e.get("app_id", "") for e in train_events if e.get("app_id")})
        self._app_vocab  = all_apps
        self._app_to_idx = {a: i + 1 for i, a in enumerate(all_apps)}  # 0 = PAD
        vocab_size = len(all_apps) + 1

        seq = [self._app_to_idx.get(e.get("app_id", ""), 0) for e in train_events]
        X = torch.tensor(seq[:-1], dtype=torch.long).unsqueeze(0)  # (1, T-1)
        y = torch.tensor(seq[1:],  dtype=torch.long).unsqueeze(0)  # (1, T-1)

        self._model = self._build_model(vocab_size)
        optimizer  = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion  = nn.CrossEntropyLoss(ignore_index=0)
        CHUNK      = 256  # BPTT chunk size

        self._model.train()
        for _ in _try_tqdm(range(self.n_epochs), desc="LSTM training", unit="epoch", leave=False):
            for i in range(0, X.shape[1] - CHUNK, CHUNK):
                x_chunk = X[:, i : i + CHUNK]
                y_chunk = y[:, i : i + CHUNK]
                optimizer.zero_grad()
                logits, _ = self._model(x_chunk)
                loss = criterion(logits.view(-1, vocab_size), y_chunk.view(-1))
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

        logger.info(f"LSTMPolicy: trained. steps={len(seq):,}  vocab={vocab_size}.")

        # -- Save cache -------------------------------------------------------
        try:
            torch.save(self._model.state_dict(), pt_path)
            _save_pkl(meta_path, {"app_vocab": self._app_vocab, "app_to_idx": self._app_to_idx})
        except Exception as exc:
            logger.warning(f"LSTMPolicy: cache save failed: {exc}")

    def predict_next_apps(self, current_app: str, context: dict) -> List[str]:
        if self._model is None or not self._torch_available:
            return [app for app, _ in self._freq.most_common(self.top_k)]

        import torch

        seq = list(self._history)
        curr_idx = self._app_to_idx.get(current_app, 0)
        if not seq or seq[-1] != curr_idx:
            seq.append(curr_idx)
        seq = seq[-20:]
        x = torch.tensor([seq], dtype=torch.long)

        self._model.eval()
        with torch.no_grad():
            logits, _ = self._model(x)
            probs = torch.softmax(logits[0, -1, :], dim=-1)
            top_indices = torch.topk(probs, min(self.top_k + 1, len(probs))).indices.tolist()

        idx_to_app = {v: k for k, v in self._app_to_idx.items()}
        results: List[str] = []
        for idx in top_indices:
            app = idx_to_app.get(idx, "")
            if app and app != current_app:
                results.append(app)
            if len(results) >= self.top_k:
                break

        # Pad with frequency fallback if fewer than top_k
        for app, _ in self._freq.most_common():
            if len(results) >= self.top_k:
                break
            if app not in results:
                results.append(app)

        return results[: self.top_k]

    def update(self, event: dict) -> None:
        app_id = event.get("app_id", "")
        self._freq[app_id] += 1
        if app_id:
            idx = self._app_to_idx.get(app_id, 0)
            self._history.append(idx)
            if len(self._history) > 20:
                self._history.pop(0)


# ---------------------------------------------------------------------------
# Prophet Baseline
# ---------------------------------------------------------------------------

class ProphetPolicy:
    """
    Facebook Prophet per-app usage forecasting baseline.

    Speed optimisation: only fits Prophet on the top 300 most-frequent apps.
    All remaining apps receive a mean-count estimate as fallback
    (no Prophet overhead for the long tail).

    Caching: forecasts are persisted to models/saved/prophet_{tag}.pkl so
    subsequent runs load in milliseconds.
    """

    _MAX_PROPHET_APPS: int = 300

    def __init__(self, top_k: int = 8) -> None:
        self.top_k = top_k
        self._forecasts: Dict[str, float] = {}
        self._freq: Counter = Counter()
        self._prophet_available = False

        for _pkg in ("prophet", "fbprophet"):
            try:
                _mod = __import__(_pkg)
                self._Prophet = _mod.Prophet
                self._prophet_available = True
                break
            except ImportError:
                pass
        if not self._prophet_available:
            logger.warning("ProphetPolicy: prophet not installed -- frequency fallback active.")

    def get_name(self) -> str:
        return "Prophet"

    def reset(self) -> None:
        self._forecasts.clear()
        self._freq.clear()

    def train(self, events: list) -> None:
        """
        Build per-app Prophet forecasts from training events.
        Loads from cache if available; saves to cache after training.
        """
        tag = _dataset_tag(len(events))
        cache_path = _pkl_path("prophet", tag)

        # -- Try cache first --------------------------------------------------
        cached = _load_pkl(cache_path)
        if cached is not None:
            self._forecasts = cached["forecasts"]
            self._freq = Counter(cached["freq"])
            logger.info(f"ProphetPolicy: {len(self._forecasts)} forecasts loaded from cache.")
            return

        # -- Train from scratch -----------------------------------------------
        self.reset()
        for event in events:
            self._freq[event.get("app_id", "")] += 1

        if not self._prophet_available:
            logger.info("ProphetPolicy: frequency fallback (Prophet unavailable).")
            _save_pkl(cache_path, {"forecasts": {}, "freq": dict(self._freq)})
            return

        import pandas as pd
        import logging as _log
        _log.getLogger("prophet").setLevel(_log.ERROR)
        _log.getLogger("cmdstanpy").setLevel(_log.ERROR)

        # Only fit top N apps -- use mean for the rest
        top_apps = {app for app, _ in self._freq.most_common(self._MAX_PROPHET_APPS)}

        # Mean-count fallback for low-frequency apps
        n_days_approx = max(1, len(events) // 80)
        for app_id, count in self._freq.items():
            if app_id not in top_apps:
                self._forecasts[app_id] = count / n_days_approx

        # Build daily usage per top app
        app_daily: Dict[str, Counter] = defaultdict(Counter)
        for i, event in enumerate(events):
            app_id = event.get("app_id", "")
            if app_id in top_apps:
                day = i // 80
                app_daily[app_id][day] += 1

        fitted = 0
        items = list(app_daily.items())
        for app_id, day_counts in _try_tqdm(items, desc="Prophet: fitting", unit="app", leave=False):
            if len(day_counts) < 3:
                self._forecasts[app_id] = float(sum(day_counts.values())) / max(1, len(day_counts))
                continue
            try:
                max_day = max(day_counts.keys())
                df = pd.DataFrame({
                    "ds": pd.date_range("2024-01-01", periods=max_day + 1, freq="D"),
                    "y":  [float(day_counts.get(d, 0)) for d in range(max_day + 1)],
                })
                m = self._Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    changepoint_prior_scale=0.1,
                )
                m.fit(df)
                future   = m.make_future_dataframe(periods=1, freq="D")
                forecast = m.predict(future)
                self._forecasts[app_id] = max(0.0, float(forecast["yhat"].iloc[-1]))
                fitted += 1
            except Exception:
                avg = float(sum(day_counts.values())) / max(1, len(day_counts))
                self._forecasts[app_id] = avg

        logger.info(f"ProphetPolicy: fitted {fitted}/{len(top_apps)} app models.")
        _save_pkl(cache_path, {"forecasts": dict(self._forecasts), "freq": dict(self._freq)})

    def predict_next_apps(self, current_app: str, context: dict) -> List[str]:
        if self._forecasts:
            ranked = sorted(self._forecasts.items(), key=lambda x: x[1], reverse=True)
            return [app for app, _ in ranked[: self.top_k]]
        return [app for app, _ in self._freq.most_common(self.top_k)]

    def update(self, event: dict) -> None:
        self._freq[event.get("app_id", "")] += 1
