"""
src/models/transformer_reranker.py

GraphMind V6 -- Transformer-based Reranker.

Architecture (v2 -- Embedding-based):
    Input:  K candidates x (app_embedding(32) + confidence(1) + time_norm(1))
            Shape: (K, 34)  -- compact, dense, generalises across apps
    Encoder: 2-layer Multi-Head Attention (4 heads, d_model=64) + LayerNorm + Dropout
    Head:   Linear(64, 1) -> squeeze -> softmax over K
    Output: Probability distribution over K candidates (sum=1)

Key improvement over v1:
    v1 used one-hot(n_apps) = 1266-dim input for real UbiqLog data.
    v2 uses nn.Embedding(n_apps, 32) = 34-dim input regardless of vocab size.
    This makes learning ~37x faster and generalises far better across users.

Per-user training:
    EmbeddingRerankerTrainer trains one small model per user from that user's
    events only, which eliminates gradient conflicts across 31 heterogeneous users.

Training:
    Loss: cross-entropy on the index of the ground-truth next app (if in top-K)
    If ground truth not in top-K: sample is skipped (no loss).
    Optimizer: Adam, lr=1e-3, cosine annealing LR decay.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Legacy v1: one-hot based (kept for backward compat / synthetic dataset)
# ---------------------------------------------------------------------------

class CandidateEncoder(nn.Module):
    """
    Encodes each candidate app into a d_model-dimensional vector.
    Input:  (batch, K, input_dim)   where input_dim = N_apps + 2
    Output: (batch, K, d_model)
    """

    def __init__(self, input_dim: int, d_model: int = 64) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TransformerReranker(nn.Module):
    """
    Legacy one-hot based transformer reranker (v1).
    Still used for synthetic dataset where vocab is small (~120 apps).
    For large real-world datasets use EmbeddingTransformerReranker (v2).
    """

    def __init__(
        self,
        n_apps: int,
        top_k: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_apps = n_apps
        self.top_k = top_k
        self.d_model = d_model

        # +2 for confidence score and time_bucket
        input_dim = n_apps + 2
        self.encoder = CandidateEncoder(input_dim, d_model)

        # Positional encoding (learned, for the K candidates)
        self.pos_embedding = nn.Embedding(top_k, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Scoring head: project each candidate to a scalar score
        self.score_head = nn.Linear(d_model, 1)

    def forward(self, candidates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            candidates: (batch, K, n_apps + 2)
                        Each row = one_hot(app) + [confidence, time_norm]

        Returns:
            scores: (batch, K) -- softmax probability over K candidates
        """
        batch_size, k, _ = candidates.shape

        # Encode candidates
        x = self.encoder(candidates)                      # (B, K, d_model)

        # Add learned positional embedding for each candidate slot
        positions = torch.arange(k, device=candidates.device).unsqueeze(0)  # (1, K)
        x = x + self.pos_embedding(positions)             # (B, K, d_model)

        # Transformer self-attention over K candidates
        x = self.transformer(x)                           # (B, K, d_model)

        # Score each candidate
        scores = self.score_head(x).squeeze(-1)           # (B, K)

        return F.softmax(scores, dim=-1)                  # (B, K)


class RerankDataset(Dataset):
    """Legacy dataset of (candidates_tensor, label) pairs for v1 reranker."""

    def __init__(
        self,
        samples: List[Tuple[torch.Tensor, int]],
    ) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.samples[idx]


def build_candidate_tensor(
    candidates: List[str],
    confidences: List[float],
    time_norm: float,
    app_vocab: List[str],
    top_k: int,
) -> torch.Tensor:
    """
    Legacy v1: Build a (top_k, n_apps + 2) one-hot feature tensor.
    Use build_candidate_indices() instead for the embedding-based v2 reranker.
    """
    n_apps = len(app_vocab)
    app_to_idx = {app: i for i, app in enumerate(app_vocab)}
    rows = []

    for i in range(top_k):
        if i < len(candidates):
            app_id = candidates[i]
            conf = float(confidences[i]) if i < len(confidences) else 0.0
            one_hot = torch.zeros(n_apps)
            if app_id in app_to_idx:
                one_hot[app_to_idx[app_id]] = 1.0
            row = torch.cat([one_hot, torch.tensor([conf, time_norm])])
        else:
            # Pad with zeros for missing candidates
            row = torch.zeros(n_apps + 2)
        rows.append(row)

    return torch.stack(rows, dim=0)   # (top_k, n_apps + 2)


class RerankerTrainer:
    """Legacy v1 trainer using one-hot TransformerReranker."""

    def __init__(
        self,
        n_apps: int,
        top_k: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        lr: float = 1e-3,
        n_epochs: int = 30,
        batch_size: int = 64,
        device: str = "cpu",
    ) -> None:
        self.model = TransformerReranker(n_apps, top_k, d_model, n_heads, n_layers)
        self.model = self.model.to(device)
        self.device = device
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min=lr * 0.01
        )

    def train(self, samples: List[Tuple[torch.Tensor, int]]) -> List[float]:
        if not samples:
            logger.warning("RerankerTrainer.train(): no samples provided.")
            return []

        dataset = RerankDataset(samples)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        epoch_losses = []

        for epoch in range(self.n_epochs):
            total_loss = 0.0
            n_batches = 0
            for batch_candidates, batch_labels in loader:
                batch_candidates = batch_candidates.to(self.device).float()
                batch_labels = batch_labels.to(self.device).long()

                self.optimizer.zero_grad()
                scores = self.model(batch_candidates)          # (B, K)
                loss = F.cross_entropy(scores, batch_labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(1, n_batches)
            epoch_losses.append(avg_loss)
            self.scheduler.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"  Reranker epoch {epoch+1}/{self.n_epochs} -- loss={avg_loss:.4f}")

        return epoch_losses

    def evaluate(self, samples: List[Tuple[torch.Tensor, int]]) -> dict:
        if not samples:
            return {"hit_at_1": 0.0, "hit_at_3": 0.0, "n_samples": 0}

        self.model.eval()
        hit1 = hit3 = 0
        with torch.no_grad():
            for candidates_tensor, label in samples:
                candidates_tensor = candidates_tensor.unsqueeze(0).to(self.device).float()
                scores = self.model(candidates_tensor).squeeze(0)  # (K,)
                top3 = torch.topk(scores, min(3, len(scores))).indices.tolist()
                if label == top3[0]:
                    hit1 += 1
                if label in top3:
                    hit3 += 1

        n = len(samples)
        return {
            "hit_at_1": round(hit1 / n * 100, 2),
            "hit_at_3": round(hit3 / n * 100, 2),
            "n_samples": n,
        }

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
        logger.info(f"Reranker model saved -> {path}")

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Reranker model loaded <- {path}")


# ---------------------------------------------------------------------------
# v2: Embedding-based reranker (34-dim input vs 1266-dim for UbiqLog)
# ---------------------------------------------------------------------------

def build_candidate_indices(
    candidates: List[str],
    confidences: List[float],
    time_norm: float,
    app_to_idx: Dict[str, int],
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build compact (app_indices, extra_features) tensors for the v2 reranker.

    Args:
        candidates:  Ordered list of candidate app IDs (length <= top_k).
        confidences: Confidence scores for each candidate.
        time_norm:   Normalised time bucket [0, 1].
        app_to_idx:  Mapping from app_id -> 1-based integer index (0 = PAD).
        top_k:       Target number of candidates (pad with 0 if fewer).

    Returns:
        app_indices:   (top_k,) LongTensor of app indices
        extra_features:(top_k, 2) FloatTensor of [confidence, time_norm]
    """
    app_indices = []
    extra_features = []

    for i in range(top_k):
        if i < len(candidates):
            idx = app_to_idx.get(candidates[i], 0)
            conf = float(confidences[i]) if i < len(confidences) else 0.0
        else:
            idx = 0
            conf = 0.0
        app_indices.append(idx)
        extra_features.append([conf, time_norm])

    return (
        torch.tensor(app_indices, dtype=torch.long),
        torch.tensor(extra_features, dtype=torch.float32),
    )


class EmbeddingTransformerReranker(nn.Module):
    """
    v2 embedding-based transformer reranker.

    Key difference vs v1: uses nn.Embedding(n_apps, embed_dim) instead of
    one-hot encoding, giving a 34-dim dense input vs 1268-dim sparse input.
    This is ~37x smaller for UbiqLog (1266 apps) and learns far faster.

    Parameters:
        n_apps:    Vocabulary size (number of unique apps). Embedding = [1, n_apps].
        top_k:     Number of candidates (K) to rerank.
        embed_dim: App embedding dimension (default 32).
        d_model:   Transformer hidden dimension (default 64).
        n_heads:   Number of attention heads (default 4).
        n_layers:  Number of transformer encoder layers (default 2).
        dropout:   Dropout probability (default 0.1).
    """

    def __init__(
        self,
        n_apps: int,
        top_k: int = 8,
        embed_dim: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_apps = n_apps
        self.top_k = top_k

        # App embedding lookup (index 0 = padding / unknown)
        self.app_embedding = nn.Embedding(n_apps + 1, embed_dim, padding_idx=0)

        # Project (embed_dim + 2) -> d_model
        # +2 for confidence score and normalised time bucket
        self.proj = nn.Sequential(
            nn.Linear(embed_dim + 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Learned positional embedding for K candidate slots
        self.pos_embedding = nn.Embedding(top_k, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Scoring head: project each candidate to a scalar score
        self.score_head = nn.Linear(d_model, 1)

    def forward(
        self,
        app_indices: torch.Tensor,
        extra_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            app_indices:    (batch, K) LongTensor of app indices.
            extra_features: (batch, K, 2) FloatTensor of [confidence, time_norm].

        Returns:
            scores: (batch, K) -- softmax probability over K candidates.
        """
        emb = self.app_embedding(app_indices)          # (B, K, embed_dim)
        x = torch.cat([emb, extra_features], dim=-1)   # (B, K, embed_dim+2)
        x = self.proj(x)                               # (B, K, d_model)

        pos = torch.arange(self.top_k, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(pos)               # (B, K, d_model)

        x = self.transformer(x)                        # (B, K, d_model)
        scores = self.score_head(x).squeeze(-1)        # (B, K)

        return F.softmax(scores, dim=-1)               # (B, K)


class EmbeddingRerankDataset(Dataset):
    """Dataset of (app_indices, extra_features, label) triples for v2 reranker."""

    def __init__(
        self,
        samples: List[Tuple[torch.Tensor, torch.Tensor, int]],
    ) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return self.samples[idx]


class EmbeddingRerankerTrainer:
    """
    Trains an EmbeddingTransformerReranker per user on compact index-based features.

    Designed for multi-user real-world datasets (e.g., UbiqLog with 31 users
    and 1266 unique apps).  One trainer instance is created per user so that
    the model only sees a single user's consistent app-usage patterns and
    converges in a handful of epochs.
    """

    def __init__(
        self,
        n_apps: int,
        top_k: int = 8,
        embed_dim: int = 32,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        lr: float = 1e-3,
        n_epochs: int = 10,
        batch_size: int = 128,
        device: str = "cpu",
    ) -> None:
        self.n_apps = n_apps
        self.top_k = top_k
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

        self.model = EmbeddingTransformerReranker(
            n_apps=n_apps,
            top_k=top_k,
            embed_dim=embed_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, n_epochs), eta_min=lr * 0.01
        )

    def train(
        self,
        samples: List[Tuple[torch.Tensor, torch.Tensor, int]],
        user_label: str = "",
    ) -> List[float]:
        """
        Train on a list of (app_indices, extra_features, label) triples.

        Returns:
            List of per-epoch average loss values.
        """
        if not samples:
            logger.warning(f"EmbeddingRerankerTrainer.train(): no samples{' for ' + user_label if user_label else ''}.")
            return []

        dataset = EmbeddingRerankDataset(samples)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        epoch_losses: List[float] = []

        for epoch in range(self.n_epochs):
            total_loss = 0.0
            n_batches = 0

            for app_indices, extra_features, labels in loader:
                app_indices = app_indices.to(self.device).long()
                extra_features = extra_features.to(self.device).float()
                labels = labels.to(self.device).long()

                self.optimizer.zero_grad()
                scores = self.model(app_indices, extra_features)  # (B, K)
                loss = F.cross_entropy(scores, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(1, n_batches)
            epoch_losses.append(avg_loss)
            self.scheduler.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                tag = f" [{user_label}]" if user_label else ""
                logger.info(
                    f"  Reranker{tag} epoch {epoch+1}/{self.n_epochs} -- loss={avg_loss:.4f}"
                )

        return epoch_losses

    def evaluate(
        self,
        samples: List[Tuple[torch.Tensor, torch.Tensor, int]],
    ) -> dict:
        """
        Evaluate Hit@1 and Hit@3 on test samples.

        Returns:
            dict with 'hit_at_1', 'hit_at_3', 'n_samples'.
        """
        if not samples:
            return {"hit_at_1": 0.0, "hit_at_3": 0.0, "n_samples": 0}

        self.model.eval()
        hit1 = hit3 = 0

        with torch.no_grad():
            for app_indices, extra_features, label in samples:
                app_indices = app_indices.unsqueeze(0).to(self.device)
                extra_features = extra_features.unsqueeze(0).to(self.device)
                scores = self.model(app_indices, extra_features).squeeze(0)  # (K,)
                top3 = torch.topk(scores, min(3, scores.shape[0])).indices.tolist()
                if label == top3[0]:
                    hit1 += 1
                if label in top3:
                    hit3 += 1

        n = len(samples)
        return {
            "hit_at_1": round(hit1 / n * 100, 2),
            "hit_at_3": round(hit3 / n * 100, 2),
            "n_samples": n,
        }

    def rerank(
        self,
        candidates: List[str],
        app_to_idx: Dict[str, int],
        confidences: List[float],
        time_norm: float,
    ) -> List[str]:
        """
        Rerank a list of candidate app IDs using the trained model.

        Returns:
            Reranked list (best candidate first).
        """
        if not candidates:
            return candidates

        app_indices, extra_features = build_candidate_indices(
            candidates=candidates,
            confidences=confidences,
            time_norm=time_norm,
            app_to_idx=app_to_idx,
            top_k=self.top_k,
        )

        self.model.eval()
        with torch.no_grad():
            app_idx_t = app_indices.unsqueeze(0).to(self.device)
            extra_t = extra_features.unsqueeze(0).to(self.device)
            scores = self.model(app_idx_t, extra_t).squeeze(0)
            ranked_indices = torch.argsort(scores, descending=True).tolist()

        reranked = []
        for idx in ranked_indices:
            if idx < len(candidates):
                reranked.append(candidates[idx])
        # Ensure all originals are present (safety)
        for c in candidates:
            if c not in reranked:
                reranked.append(c)
        return reranked

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
        logger.info(f"EmbeddingReranker saved -> {path}")

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        logger.info(f"EmbeddingReranker loaded <- {path}")
