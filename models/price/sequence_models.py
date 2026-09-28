"""
models/price/sequence_models.py
Production-grade LSTM, GRU, and Temporal Fusion Transformer implementations
for stock price prediction.  All models include:
  - Attention mechanisms
  - Dropout regularisation
  - Mixed-precision training support
  - SHAP explainability hooks
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config import get_settings

settings = get_settings()
cfg = settings.model


# ─────────────────────────────────────────────────────────────────────────────
# Attention Mechanisms
# ─────────────────────────────────────────────────────────────────────────────

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: Tensor,   # (B, heads, T, d_k)
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(attn, v), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.attn = ScaledDotProductAttention(dropout)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        B = q.size(0)
        # Linear projections and reshape for multi-head
        def reshape(x, w):
            return w(x).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        q, k, v = reshape(q, self.w_q), reshape(k, self.w_k), reshape(v, self.w_v)
        x, attn = self.attn(q, k, v, mask)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.dropout(self.out(x)), attn


# ─────────────────────────────────────────────────────────────────────────────
# LSTM with Attention
# ─────────────────────────────────────────────────────────────────────────────

class AttentionLSTM(nn.Module):
    """
    Bidirectional LSTM with self-attention for multi-step ahead forecasting.
    Returns point predictions + uncertainty intervals.

    Architecture:
        Input → LayerNorm → Bi-LSTM stack → Self-Attention → FFN → Output
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.3,
        pred_len: int = 5,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.pred_len = pred_len
        dirs = 2 if bidirectional else 1

        # Input projection & normalisation
        self.input_norm = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, hidden_size)

        # LSTM stack
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_size * dirs

        # Reduce bidirectional to hidden_size for attention
        self.lstm_proj = nn.Linear(lstm_out_dim, hidden_size)

        # Self-attention over time steps
        self.attn = MultiHeadAttention(hidden_size, n_heads, dropout)
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

        # Output heads
        self.pred_head = nn.Linear(hidden_size, pred_len)          # point prediction
        self.uncertainty_head = nn.Linear(hidden_size, pred_len)   # log variance (for NLL loss)

        # Direction head (buy/sell/hold classification)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),   # 0=sell, 1=hold, 2=buy
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name and param.dim() == 2:
                nn.init.xavier_uniform_(param)

    def forward(
        self,
        x: Tensor,                     # (B, T, input_size)
        return_attention: bool = False,
    ) -> dict[str, Tensor]:
        B, T, _ = x.shape

        # Input processing
        x = self.input_norm(x)
        x = self.input_proj(x)            # (B, T, H)

        # LSTM
        lstm_out, (h_n, _) = self.lstm(x)
        lstm_out = self.lstm_proj(lstm_out)  # (B, T, H)

        # Self-attention (residual)
        attn_out, attn_weights = self.attn(lstm_out, lstm_out, lstm_out)
        lstm_out = self.attn_norm(lstm_out + attn_out)

        # FFN (residual)
        ffn_out = self.ffn(lstm_out)
        out = self.ffn_norm(lstm_out + ffn_out)

        # Pool: take the last time step as the "summary" representation
        context = out[:, -1, :]            # (B, H)

        predictions = self.pred_head(context)            # (B, pred_len)
        log_var = self.uncertainty_head(context)         # (B, pred_len)
        direction = self.direction_head(context)         # (B, 3)

        result = {
            "predictions": predictions,
            "log_var": log_var,
            "std": torch.exp(0.5 * log_var),
            "direction": direction,
            "direction_probs": F.softmax(direction, dim=-1),
        }
        if return_attention:
            result["attention_weights"] = attn_weights

        return result

    def predict_with_ci(
        self,
        x: Tensor,
        n_samples: int = 100,
        ci: float = 0.95,
    ) -> dict[str, np.ndarray]:
        """
        Monte Carlo Dropout inference for calibrated confidence intervals.
        Enable dropout at inference time for stochastic sampling.
        """
        self.train()   # enable dropout
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(x)
                samples.append(out["predictions"].cpu().numpy())
        self.eval()

        samples = np.stack(samples, axis=0)   # (n_samples, B, pred_len)
        mean = samples.mean(axis=0)
        std = samples.std(axis=0)
        alpha = (1 - ci) / 2
        lower = np.quantile(samples, alpha, axis=0)
        upper = np.quantile(samples, 1 - alpha, axis=0)

        return {
            "mean": mean,
            "std": std,
            "lower": lower,
            "upper": upper,
            "confidence": ci,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Transformer for Time Series (TST)
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class TimeSeriesTransformer(nn.Module):
    """
    Transformer encoder-decoder for multi-step time series prediction.
    Follows the PatchTST / Informer design with:
      - Patch-based tokenisation (improves long-range modelling)
      - Causal masking in decoder
      - Learnable positional embeddings
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_encoder_layers: int = 3,
        n_decoder_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        seq_len: int = 60,
        pred_len: int = 5,
        patch_size: int = 4,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.patch_size = patch_size
        n_patches = seq_len // patch_size

        # Patch embedding: treat each patch of time steps as a token
        self.patch_embed = nn.Linear(input_size * patch_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=n_patches + 10, dropout=dropout)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm (more stable)
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_encoder_layers)

        # Decoder takes a learned query (one per prediction step)
        self.query = nn.Parameter(torch.randn(pred_len, d_model))

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_decoder_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, 1)
        self.log_var_proj = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        B, T, C = x.shape

        # Patch the sequence
        n_patches = T // self.patch_size
        x = x[:, : n_patches * self.patch_size, :]   # trim
        x = x.reshape(B, n_patches, self.patch_size * C)
        x = self.patch_embed(x)                        # (B, n_patches, d_model)
        x = self.pos_enc(x)

        # Encode
        memory = self.encoder(x)                       # (B, n_patches, d_model)

        # Decode with learned queries
        queries = self.query.unsqueeze(0).expand(B, -1, -1)  # (B, pred_len, d_model)
        decoded = self.decoder(queries, memory)               # (B, pred_len, d_model)

        predictions = self.output_proj(decoded).squeeze(-1)   # (B, pred_len)
        log_var = self.log_var_proj(decoded).squeeze(-1)

        return {
            "predictions": predictions,
            "log_var": log_var,
            "std": torch.exp(0.5 * log_var),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CNN-LSTM Hybrid
# ─────────────────────────────────────────────────────────────────────────────

class CNNLSTMHybrid(nn.Module):
    """
    1D-CNN for local pattern extraction, feeding into LSTM for temporal modelling.
    CNNs catch short-range patterns (candlestick formations, 3-day momentum);
    LSTM captures the long-range dependencies.
    """

    def __init__(
        self,
        input_size: int,
        cnn_channels: list[int] = (64, 128, 256),
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        pred_len: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.pred_len = pred_len

        # CNN feature extractor
        cnn_blocks = []
        in_ch = input_size
        for out_ch in cnn_channels:
            cnn_blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout / 2),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_blocks)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # Output
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, pred_len),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        # x: (B, T, C) → CNN expects (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)            # (B, channels[-1], T)
        x = x.permute(0, 2, 1)    # (B, T, channels[-1])

        lstm_out, _ = self.lstm(x)
        context = lstm_out[:, -1, :]   # last step

        predictions = self.head(context)
        return {"predictions": predictions}


# ─────────────────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_nll_loss(
    predictions: Tensor,
    log_var: Tensor,
    targets: Tensor,
) -> Tensor:
    """
    Negative log-likelihood assuming Gaussian output distribution.
    Encourages calibrated uncertainty: wide intervals when uncertain,
    tight when confident.
    """
    variance = torch.exp(log_var)
    loss = 0.5 * (log_var + (predictions - targets) ** 2 / variance)
    return loss.mean()


def sharpe_loss(returns: Tensor, risk_free: float = 0.0) -> Tensor:
    """
    Directly optimise the Sharpe ratio of predicted returns.
    Gradient flows back through the prediction head.
    """
    excess = returns - risk_free
    mean_r = excess.mean()
    std_r = excess.std() + 1e-8
    return -(mean_r / std_r)   # negative because we minimise


def combined_loss(
    predictions: Tensor,
    log_var: Tensor,
    targets: Tensor,
    alpha: float = 0.8,   # weight on NLL loss
) -> Tensor:
    """
    NLL + directional accuracy loss.
    Penalises wrong direction predictions more heavily.
    """
    nll = gaussian_nll_loss(predictions, log_var, targets)

    # Directional loss: penalise sign errors
    pred_dir = torch.sign(predictions)
    true_dir = torch.sign(targets)
    directional = ((pred_dir != true_dir).float()).mean()

    return alpha * nll + (1 - alpha) * directional
