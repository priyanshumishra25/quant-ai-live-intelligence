"""
Anomaly detection with autoencoders and Graph Neural Networks
for sector-correlation-aware embeddings.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. AUTOENCODER FOR ANOMALY DETECTION
# ===========================================================================

class TimeSeriesAutoencoder(nn.Module):
    """
    LSTM-based variational autoencoder.
    Reconstructs normal market behaviour; high reconstruction error ⟹ anomaly.
    Use cases: flash crashes, pump-and-dump, data errors, regime breaks.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        seq_len:    int = 30,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.seq_len   = seq_len
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.enc_lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=2, batch_first=True, dropout=dropout,
        )
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc_decode = nn.Linear(latent_dim, hidden_dim)
        self.dec_lstm  = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=2, batch_first=True, dropout=dropout,
        )
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, (h, _) = self.enc_lstm(x)
        h_last = h[-1]                     # last layer hidden state
        return self.fc_mu(h_last), self.fc_logvar(h_last)

    def reparameterise(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        if self.training:
            std = (0.5 * logvar).exp()
            return mu + std * torch.randn_like(std)
        return mu

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        h = F.relu(self.fc_decode(z))                      # (B, hidden)
        h_repeated = h.unsqueeze(1).repeat(1, seq_len, 1)  # (B, T, hidden)
        out, _ = self.dec_lstm(h_repeated)
        return self.output_proj(out)                        # (B, T, input_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z          = self.reparameterise(mu, logvar)
        x_hat      = self.decode(z, x.size(1))
        return x_hat, mu, logvar

    @staticmethod
    def vae_loss(
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1e-3,
    ) -> torch.Tensor:
        recon = F.mse_loss(x_hat, x, reduction="mean")
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + kl_weight * kl


class AnomalyDetector:
    """
    Wraps the VAE for training and anomaly scoring.

    Anomaly score = reconstruction error (MSE per sample).
    Threshold is set at training-set 99th percentile.
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        seq_len:    int = 30,
        device:     str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.seq_len = seq_len
        self.device  = device
        self.scaler  = StandardScaler()
        self.threshold: float = 0.0
        self.model = TimeSeriesAutoencoder(
            input_dim, hidden_dim, latent_dim, seq_len
        ).to(device)

    # -----------------------------------------------------------------------

    def _to_sequences(self, X: np.ndarray) -> torch.Tensor:
        """Slide a window of `seq_len` over the feature matrix."""
        seqs = []
        for i in range(len(X) - self.seq_len + 1):
            seqs.append(X[i : i + self.seq_len])
        return torch.tensor(np.array(seqs), dtype=torch.float32)

    # -----------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 128,
        threshold_pct: float = 99.0,
    ) -> None:
        X_sc = self.scaler.fit_transform(X.values).astype(np.float32)
        seqs = self._to_sequences(X_sc).to(self.device)

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(seqs),
            batch_size=batch_size,
            shuffle=True,
        )
        optimiser = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for (batch,) in loader:
                optimiser.zero_grad()
                x_hat, mu, logvar = self.model(batch)
                loss = TimeSeriesAutoencoder.vae_loss(batch, x_hat, mu, logvar)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimiser.step()
                total_loss += loss.item()
            scheduler.step()
            if epoch % 10 == 0:
                logger.debug("Autoencoder epoch %d loss=%.5f", epoch, total_loss / len(loader))

        # Compute training-set reconstruction errors → threshold
        errors = self._reconstruction_errors(seqs)
        self.threshold = float(np.percentile(errors, threshold_pct))
        logger.info("Anomaly threshold (p%.0f): %.5f", threshold_pct, self.threshold)

    @torch.no_grad()
    def _reconstruction_errors(self, seqs: torch.Tensor) -> np.ndarray:
        self.model.eval()
        errors = []
        for i in range(0, len(seqs), 256):
            batch = seqs[i : i + 256]
            x_hat, _, _ = self.model(batch)
            err = F.mse_loss(x_hat, batch, reduction="none").mean(dim=(1, 2))
            errors.extend(err.cpu().numpy())
        return np.array(errors)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Returns per-window reconstruction error."""
        X_sc = self.scaler.transform(X.values).astype(np.float32)
        seqs = self._to_sequences(X_sc).to(self.device)
        return self._reconstruction_errors(seqs)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns boolean anomaly flags."""
        return self.score(X) > self.threshold


# ===========================================================================
# 2. GRAPH NEURAL NETWORK — sector-aware stock embeddings
# ===========================================================================

class GATLayer(nn.Module):
    """
    Single Graph Attention (GAT) layer.
    Computes multi-head attention over node neighbours,
    weighted by edge features (correlation strength).
    """

    def __init__(
        self,
        in_dim:   int,
        out_dim:  int,
        n_heads:  int = 4,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = out_dim // n_heads
        assert out_dim % n_heads == 0

        self.W    = nn.Linear(in_dim,        out_dim, bias=False)
        self.attn = nn.Linear(2 * out_dim,   n_heads, bias=False)
        self.drop = nn.Dropout(dropout)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(
        self,
        x: torch.Tensor,          # (N, in_dim)
        adj: torch.Tensor,         # (N, N) weighted adjacency
    ) -> torch.Tensor:
        N = x.size(0)
        h = self.W(x)              # (N, out_dim)

        # Pairwise attention logits
        hi = h.unsqueeze(1).repeat(1, N, 1)   # (N, N, out_dim)
        hj = h.unsqueeze(0).repeat(N, 1, 1)   # (N, N, out_dim)
        e  = self.leaky(self.attn(torch.cat([hi, hj], dim=-1)))  # (N, N, heads)
        e  = e.mean(dim=-1)        # (N, N) average over heads

        # Mask non-edges (keep only connected pairs)
        mask = (adj == 0)
        e = e.masked_fill(mask, float("-inf"))
        alpha = F.softmax(e, dim=-1)           # (N, N) attention weights
        alpha = self.drop(alpha)

        # Weighted aggregation
        out = torch.matmul(alpha, h)           # (N, out_dim)
        return F.elu(out)


class StockGNN(nn.Module):
    """
    2-layer GAT for learning stock embeddings from sector correlation graphs.

    Input:
        x   : (N, F)   — per-stock feature vectors
        adj : (N, N)   — correlation matrix (or binary sector adjacency)

    Output:
        embeddings : (N, embed_dim)
        predictions: (N, 2) — [predicted_return, direction_prob]
    """

    def __init__(
        self,
        in_dim:    int,
        hidden:    int = 64,
        embed_dim: int = 32,
        n_heads:   int = 4,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.gat1 = GATLayer(in_dim,  hidden,    n_heads, dropout)
        self.gat2 = GATLayer(hidden,  embed_dim, n_heads, dropout)
        self.drop = nn.Dropout(dropout)
        self.pred_head = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2),    # [return_logit, direction_logit]
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.drop(self.gat1(x, adj))
        emb = self.gat2(h, adj)
        preds = self.pred_head(emb)
        return emb, preds


# ---------------------------------------------------------------------------
# Correlation adjacency builder
# ---------------------------------------------------------------------------

def build_correlation_adjacency(
    returns: pd.DataFrame,
    threshold: float = 0.3,
    window: int = 60,
) -> np.ndarray:
    """
    Builds a thresholded correlation adjacency matrix from rolling returns.
    Tickers are the columns of `returns`.
    """
    corr = returns.tail(window).corr().values
    adj  = np.where(np.abs(corr) > threshold, np.abs(corr), 0.0)
    np.fill_diagonal(adj, 0.0)   # no self-loops
    return adj.astype(np.float32)


# ---------------------------------------------------------------------------
# GNN trainer
# ---------------------------------------------------------------------------

class GNNTrainer:
    """
    Trains StockGNN on a cross-sectional feature matrix + return labels.

    At each training step:
    - `X`:      (N_stocks, F) feature matrix for today
    - `adj`:    (N_stocks, N_stocks) correlation adjacency
    - `y_ret`:  (N_stocks,) next-day returns (regression target)
    - `y_dir`:  (N_stocks,) direction labels (0/1)
    """

    def __init__(
        self,
        in_dim:    int,
        hidden:    int = 64,
        embed_dim: int = 32,
        lr:        float = 1e-3,
        device:    str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.model  = StockGNN(in_dim, hidden, embed_dim).to(device)
        self.opt    = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scaler = StandardScaler()

    def _prepare(
        self,
        X:     np.ndarray,
        adj:   np.ndarray,
        y_ret: np.ndarray,
        y_dir: np.ndarray,
    ) -> tuple[torch.Tensor, ...]:
        x_t   = torch.tensor(X,     dtype=torch.float32).to(self.device)
        adj_t = torch.tensor(adj,   dtype=torch.float32).to(self.device)
        yr_t  = torch.tensor(y_ret, dtype=torch.float32).to(self.device)
        yd_t  = torch.tensor(y_dir, dtype=torch.long).to(self.device)
        return x_t, adj_t, yr_t, yd_t

    def train_step(
        self,
        X:     np.ndarray,
        adj:   np.ndarray,
        y_ret: np.ndarray,
        y_dir: np.ndarray,
    ) -> float:
        self.model.train()
        x_t, adj_t, yr_t, yd_t = self._prepare(X, adj, y_ret, y_dir)
        self.opt.zero_grad()
        _, preds = self.model(x_t, adj_t)

        reg_loss = F.mse_loss(preds[:, 0], yr_t)
        cls_loss = F.cross_entropy(preds[:, 1:], yd_t)
        loss = reg_loss + cls_loss
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return loss.item()

    @torch.no_grad()
    def predict(
        self,
        X: np.ndarray,
        adj: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (predicted_returns, direction_probs)."""
        self.model.eval()
        x_t   = torch.tensor(X,   dtype=torch.float32).to(self.device)
        adj_t = torch.tensor(adj, dtype=torch.float32).to(self.device)
        emb, preds = self.model(x_t, adj_t)
        returns = preds[:, 0].cpu().numpy()
        dir_prob = F.softmax(preds[:, 1:], dim=-1)[:, 1].cpu().numpy()
        return returns, dir_prob

    @torch.no_grad()
    def embeddings(self, X: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """Returns node embeddings for downstream use."""
        self.model.eval()
        x_t   = torch.tensor(X,   dtype=torch.float32).to(self.device)
        adj_t = torch.tensor(adj, dtype=torch.float32).to(self.device)
        emb, _ = self.model(x_t, adj_t)
        return emb.cpu().numpy()
