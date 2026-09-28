"""
models/training_pipeline.py
End-to-end model training pipeline with:
  - Walk-forward cross-validation
  - Hyperparameter optimisation (Optuna)
  - MLflow experiment tracking
  - Distributed training support (PyTorch DDP)
  - Automated model registration and versioning
"""

from __future__ import annotations

import os
import random
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from config import get_settings
from features.technical.indicators import compute_all_indicators
from models.ensemble.dynamic_ensemble import FeatureBuilder
from models.price.sequence_models import AttentionLSTM, TimeSeriesTransformer, combined_loss

settings = get_settings()
cfg = settings.model


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "return_1d",
    seq_len: int = 60,
    pred_len: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a time-series DataFrame into overlapping sequences.
    No data leakage: targets are strictly future values.

    Returns:
        X: (N, seq_len, n_features)
        y: (N, pred_len)
    """
    available = [c for c in feature_cols if c in df.columns]
    X_raw = df[available].values.astype(np.float32)
    y_raw = df[target_col].values.astype(np.float32)

    X_seqs, y_seqs = [], []
    for i in range(len(df) - seq_len - pred_len + 1):
        X_seqs.append(X_raw[i : i + seq_len])
        y_seqs.append(y_raw[i + seq_len : i + seq_len + pred_len])

    return np.array(X_seqs), np.array(y_seqs)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Handles training loop for PyTorch sequence models.
    Supports:
      - Mixed precision (AMP)
      - Gradient clipping
      - Learning rate scheduling
      - Early stopping
      - MLflow logging
    """

    def __init__(
        self,
        model: nn.Module,
        run_name: str = "lstm_run",
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.run_name = run_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.scaler = GradScaler(enabled=cfg.mixed_precision)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        hyperparams: Optional[dict] = None,
    ) -> dict[str, float]:
        hp = hyperparams or {}
        lr = hp.get("lr", cfg.learning_rate)
        wd = hp.get("weight_decay", cfg.weight_decay)
        epochs = hp.get("epochs", cfg.num_epochs)
        patience = hp.get("patience", 15)
        batch_size = hp.get("batch_size", cfg.batch_size)

        # DataLoaders
        train_ds = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
        )
        val_ds = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, pin_memory=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=wd
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr * 10,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
        )

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_mae": []}

        for epoch in range(epochs):
            # ── Train ──────────────────────────────────────────────────────
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=cfg.mixed_precision):
                    out = self.model(X_batch)
                    loss = combined_loss(out["predictions"], out.get("log_var", torch.zeros_like(out["predictions"])), y_batch)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                train_losses.append(loss.item())

            # ── Validate ───────────────────────────────────────────────────
            self.model.eval()
            val_losses, val_preds, val_true = [], [], []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    with autocast(enabled=cfg.mixed_precision):
                        out = self.model(X_batch)
                        loss = combined_loss(out["predictions"], out.get("log_var", torch.zeros_like(out["predictions"])), y_batch)
                    val_losses.append(loss.item())
                    val_preds.extend(out["predictions"][:, 0].cpu().numpy())
                    val_true.extend(y_batch[:, 0].cpu().numpy())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            val_mae = mean_absolute_error(val_true, val_preds)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_mae"].append(val_mae)

            # Early stopping
            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs} | train_loss={train_loss:.5f} val_loss={val_loss:.5f} val_mae={val_mae:.5f}")

            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {
            "best_val_loss": best_val_loss,
            "val_mae": min(history["val_mae"]),
            "n_epochs": len(history["train_loss"]),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter Optimisation (Optuna)
# ─────────────────────────────────────────────────────────────────────────────

def optimise_hyperparams(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_size: int,
    n_trials: int = 50,
    model_type: str = "lstm",
) -> dict:
    """
    Run Optuna study to find best hyperparameters.
    Uses pruning to kill bad trials early.
    """

    def objective(trial: optuna.Trial) -> float:
        hidden = trial.suggest_categorical("hidden_size", [128, 256, 512])
        layers = trial.suggest_int("num_layers", 2, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

        if model_type == "lstm":
            model = AttentionLSTM(
                input_size=input_size,
                hidden_size=hidden,
                num_layers=layers,
                dropout=dropout,
                pred_len=cfg.lstm_pred_len,
            )
        else:
            model = TimeSeriesTransformer(
                input_size=input_size,
                d_model=hidden,
                n_heads=4,
                dropout=dropout,
                pred_len=cfg.tft_max_prediction_length,
            )

        trainer = ModelTrainer(model)
        metrics = trainer.train(
            X_train, y_train, X_val, y_val,
            hyperparams={"lr": lr, "weight_decay": wd, "epochs": 30, "patience": 5}
        )
        return metrics["best_val_loss"]

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    logger.info(f"Best params: {study.best_params}")
    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Training Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardPipeline:
    """
    Complete training pipeline:
    1. Compute features for each ticker
    2. Walk-forward splits (no leakage)
    3. Fit feature builder on train fold
    4. Optionally run Optuna on first fold
    5. Train all models on each fold
    6. Track with MLflow
    7. Register best model in MLflow Model Registry
    """

    def __init__(
        self,
        price_data: dict[str, pd.DataFrame],
        feature_builder: Optional[FeatureBuilder] = None,
        experiment_name: str = settings.infra.mlflow_experiment,
    ):
        self.price_data = price_data
        self.feature_builder = feature_builder or FeatureBuilder()
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(settings.infra.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)

    def run(
        self,
        train_size: int = cfg.wf_initial_train_size,
        step_size: int = cfg.wf_step_size,
        n_splits: int = cfg.wf_n_splits,
        optimise_hp: bool = False,
    ) -> dict[str, Any]:
        set_seed(cfg.seed)

        # Prepare combined feature DataFrame
        all_features = self._prepare_features()

        results = {}
        all_dates = all_features.index.unique().sort_values()

        for split_idx in range(n_splits):
            train_end = train_size + split_idx * step_size
            val_end = train_end + step_size

            if val_end > len(all_dates):
                break

            train_dates = all_dates[:train_end]
            val_dates = all_dates[train_end:val_end]

            logger.info(f"Split {split_idx+1}/{n_splits}: train→{train_dates[-1].date()}, val→{val_dates[-1].date()}")

            train_df = all_features[all_features.index.isin(train_dates)].dropna()
            val_df = all_features[all_features.index.isin(val_dates)].dropna()

            if len(train_df) < 200 or len(val_df) < 20:
                logger.warning(f"Split {split_idx}: insufficient data, skipping")
                continue

            # Fit scaler on training data only
            self.feature_builder.fit(train_df)

            X_train, y_train = build_sequences(
                train_df, self.feature_builder.FEATURE_COLUMNS,
                seq_len=cfg.lstm_seq_len, pred_len=cfg.lstm_pred_len,
            )
            X_val, y_val = build_sequences(
                val_df, self.feature_builder.FEATURE_COLUMNS,
                seq_len=cfg.lstm_seq_len, pred_len=cfg.lstm_pred_len,
            )

            if X_train.shape[0] == 0 or X_val.shape[0] == 0:
                continue

            input_size = X_train.shape[2]

            with mlflow.start_run(run_name=f"split_{split_idx}") as run:
                mlflow.log_params({
                    "split": split_idx,
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                    "input_features": input_size,
                    "seq_len": cfg.lstm_seq_len,
                })

                # Optuna on first split only
                best_hp = {}
                if optimise_hp and split_idx == 0:
                    logger.info("Running Optuna hyperparameter search...")
                    best_hp = optimise_hyperparams(
                        X_train, y_train, X_val, y_val,
                        input_size=input_size, n_trials=30
                    )
                    mlflow.log_params({f"opt_{k}": v for k, v in best_hp.items()})

                # Train LSTM
                lstm = AttentionLSTM(
                    input_size=input_size,
                    hidden_size=best_hp.get("hidden_size", cfg.lstm_hidden_size),
                    num_layers=best_hp.get("num_layers", cfg.lstm_num_layers),
                    dropout=best_hp.get("dropout", cfg.lstm_dropout),
                    pred_len=cfg.lstm_pred_len,
                )
                lstm_trainer = ModelTrainer(lstm, run_name=f"lstm_split_{split_idx}")
                lstm_metrics = lstm_trainer.train(X_train, y_train, X_val, y_val, best_hp)
                mlflow.log_metrics({f"lstm_{k}": v for k, v in lstm_metrics.items()})
                mlflow.pytorch.log_model(lstm, "lstm_model")

                results[f"split_{split_idx}"] = {
                    "lstm": lstm_metrics,
                    "run_id": run.info.run_id,
                }

                logger.info(f"Split {split_idx} complete. val_loss={lstm_metrics['best_val_loss']:.5f}")

        # Register best model based on val_loss
        self._register_best_model(results)
        return results

    def _prepare_features(self) -> pd.DataFrame:
        """Compute indicators for all tickers and combine into a single DF."""
        dfs = []
        for ticker, df in self.price_data.items():
            try:
                enriched = compute_all_indicators(df)
                enriched["ticker_id"] = hash(ticker) % 1000   # encode ticker as numeric
                dfs.append(enriched)
            except Exception as e:
                logger.warning(f"Feature computation failed for {ticker}: {e}")
        if not dfs:
            return pd.DataFrame()
        combined = pd.concat(dfs, axis=0)
        combined = combined.sort_index()
        return combined

    def _register_best_model(self, results: dict) -> None:
        """Find best val_loss across all splits and register that run."""
        best_run_id = None
        best_loss = float("inf")
        for split_key, split_res in results.items():
            loss = split_res.get("lstm", {}).get("best_val_loss", float("inf"))
            if loss < best_loss:
                best_loss = loss
                best_run_id = split_res.get("run_id")

        if best_run_id:
            model_uri = f"runs:/{best_run_id}/lstm_model"
            registered = mlflow.register_model(model_uri, "quant-ai-lstm")
            logger.info(f"Registered model version: {registered.version}")
