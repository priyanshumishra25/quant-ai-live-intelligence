"""
Tree-based model training: XGBoost and LightGBM.
Walk-forward validation, Optuna hyperparameter search,
SHAP feature importance, and MLflow experiment tracking.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from optuna.integration import LightGBMPruningCallback
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardSplit:
    train_idx: np.ndarray
    val_idx:   np.ndarray
    fold:      int


def walk_forward_splits(
    n: int,
    n_folds: int = 5,
    min_train: int = 252,
    gap: int = 5,
) -> list[WalkForwardSplit]:
    """
    Generates walk-forward (expanding window) splits.
    `gap` prevents leakage from overlapping windows.
    """
    splits = []
    fold_size = (n - min_train) // n_folds
    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        val_start = train_end + gap
        val_end   = min(val_start + fold_size, n)
        if val_end <= val_start:
            break
        splits.append(WalkForwardSplit(
            train_idx=np.arange(0, train_end),
            val_idx=np.arange(val_start, val_end),
            fold=fold,
        ))
    return splits


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Percentage of correct return-sign predictions."""
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def sharpe_of_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    risk_free: float = 0.0,
) -> float:
    """Simulated daily Sharpe: go long when pred>0, short when pred<0."""
    returns = np.sign(y_pred) * y_true
    excess  = returns - risk_free / 252
    return float(excess.mean() / (excess.std() + 1e-8) * np.sqrt(252))


# ---------------------------------------------------------------------------
# XGBoost trainer
# ---------------------------------------------------------------------------

class XGBoostTrainer:
    """
    Trains XGBoost for multi-target stock prediction:
      - reg:squarederror  → next-day return
      - binary:logistic   → directional probability
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "objective":        "reg:squarederror",
        "n_estimators":     800,
        "learning_rate":    0.03,
        "max_depth":        5,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "tree_method":      "hist",
        "device":           "cuda",    # falls back to cpu silently
        "random_state":     42,
        "n_jobs":           -1,
    }

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.reg_model: xgb.XGBRegressor | None = None
        self.cls_model: xgb.XGBClassifier | None = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []

    # --- Optuna objective ---------------------------------------------------

    def _xgb_objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        params = {
            "objective":        "reg:squarederror",
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate":    trial.suggest_float("lr", 1e-3, 0.1, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":        trial.suggest_float("alpha", 0.0, 2.0),
            "reg_lambda":       trial.suggest_float("lambda", 0.0, 5.0),
            "tree_method":      "hist",
            "device":           "cuda",
            "random_state":     42,
        }
        model = xgb.XGBRegressor(**params, early_stopping_rounds=50, verbosity=0)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_hat = model.predict(X_val)
        return float(mean_squared_error(y_val, y_hat, squared=False))

    def optimise(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
        val_frac: float = 0.2,
    ) -> dict[str, Any]:
        split = int(len(X) * (1 - val_frac))
        X_tr, X_v = X[:split], X[split:]
        y_tr, y_v = y[:split], y[split:]

        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda t: self._xgb_objective(t, X_tr, y_tr, X_v, y_v),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        logger.info("XGBoost best RMSE: %.5f", study.best_value)
        best = study.best_params
        best["objective"]    = "reg:squarederror"
        best["tree_method"]  = "hist"
        best["device"]       = "cuda"
        best["random_state"] = 42
        return best

    # --- Train ---------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y_return: pd.Series,
        y_direction: pd.Series,
        eval_set_frac: float = 0.1,
    ) -> None:
        self.feature_names = list(X.columns)
        X_sc = self.scaler.fit_transform(X.values)
        split = int(len(X_sc) * (1 - eval_set_frac))

        # Regression model
        self.reg_model = xgb.XGBRegressor(
            **{**self.params, "objective": "reg:squarederror"},
            early_stopping_rounds=50,
            verbosity=0,
        )
        self.reg_model.fit(
            X_sc[:split], y_return.values[:split],
            eval_set=[(X_sc[split:], y_return.values[split:])],
            verbose=False,
        )

        # Classification model (direction)
        cls_params = {**self.params, "objective": "binary:logistic"}
        self.cls_model = xgb.XGBClassifier(**cls_params, early_stopping_rounds=50, verbosity=0)
        self.cls_model.fit(
            X_sc[:split], y_direction.values[:split],
            eval_set=[(X_sc[split:], y_direction.values[split:])],
            verbose=False,
        )

    # --- Predict -------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        X_sc = self.scaler.transform(X.values)
        assert self.reg_model and self.cls_model
        return {
            "return":    self.reg_model.predict(X_sc),
            "direction": self.cls_model.predict_proba(X_sc)[:, 1],
        }

    # --- Walk-forward evaluation ---------------------------------------------

    def walk_forward_eval(
        self,
        X: pd.DataFrame,
        y_return: pd.Series,
        y_direction: pd.Series,
        n_folds: int = 5,
    ) -> pd.DataFrame:
        results = []
        splits = walk_forward_splits(len(X), n_folds=n_folds)
        for split in splits:
            X_tr = X.iloc[split.train_idx]
            y_r_tr = y_return.iloc[split.train_idx]
            y_d_tr = y_direction.iloc[split.train_idx]

            X_v = X.iloc[split.val_idx]
            y_r_v = y_return.iloc[split.val_idx]
            y_d_v = y_direction.iloc[split.val_idx]

            self.fit(X_tr, y_r_tr, y_d_tr)
            preds = self.predict(X_v)

            rmse  = mean_squared_error(y_r_v, preds["return"], squared=False)
            da    = directional_accuracy(y_r_v.values, preds["return"])
            sharpe = sharpe_of_predictions(y_r_v.values, preds["return"])
            auc   = roc_auc_score(y_d_v.values, preds["direction"])

            results.append({
                "fold": split.fold, "rmse": rmse,
                "dir_acc": da, "sharpe": sharpe, "auc": auc,
            })
            logger.info(
                "XGB fold %d | RMSE=%.5f DA=%.1f%% Sharpe=%.2f AUC=%.3f",
                split.fold, rmse, da * 100, sharpe, auc,
            )
        return pd.DataFrame(results)

    # --- SHAP ----------------------------------------------------------------

    def shap_importance(
        self,
        X: pd.DataFrame,
        n_samples: int = 500,
    ) -> pd.DataFrame:
        X_sc = self.scaler.transform(X.values[:n_samples])
        explainer = shap.TreeExplainer(self.reg_model)
        values = explainer.shap_values(X_sc)
        importance = np.abs(values).mean(axis=0)
        return pd.DataFrame({
            "feature":    self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    # --- Save / load ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.reg_model:
            self.reg_model.save_model(str(path / "xgb_reg.json"))
        if self.cls_model:
            self.cls_model.save_model(str(path / "xgb_cls.json"))
        joblib.dump(self.scaler, path / "scaler.pkl")
        joblib.dump(self.feature_names, path / "features.pkl")
        logger.info("XGBoost models saved to %s", path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self.reg_model = xgb.XGBRegressor()
        self.reg_model.load_model(str(path / "xgb_reg.json"))
        self.cls_model = xgb.XGBClassifier()
        self.cls_model.load_model(str(path / "xgb_cls.json"))
        self.scaler       = joblib.load(path / "scaler.pkl")
        self.feature_names = joblib.load(path / "features.pkl")


# ---------------------------------------------------------------------------
# LightGBM trainer
# ---------------------------------------------------------------------------

class LightGBMTrainer:
    """
    LightGBM for return regression + direction classification.
    Much faster than XGBoost; also supports categorical features natively.
    """

    DEFAULT_PARAMS: dict[str, Any] = {
        "objective":        "regression",
        "metric":           "rmse",
        "num_leaves":       63,
        "learning_rate":    0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "min_child_samples": 20,
        "reg_alpha":        0.1,
        "reg_lambda":       1.0,
        "n_estimators":     1000,
        "device_type":      "gpu",    # falls back gracefully
        "verbose":          -1,
        "random_state":     42,
        "n_jobs":           -1,
    }

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.reg_model: lgb.LGBMRegressor | None = None
        self.cls_model: lgb.LGBMClassifier | None = None
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []

    # --- Optuna objective ---------------------------------------------------

    def _lgb_objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        params = {
            "objective":         "regression",
            "metric":            "rmse",
            "num_leaves":        trial.suggest_int("num_leaves", 20, 200),
            "learning_rate":     trial.suggest_float("lr", 1e-3, 0.1, log=True),
            "feature_fraction":  trial.suggest_float("ff", 0.4, 1.0),
            "bagging_fraction":  trial.suggest_float("bf", 0.4, 1.0),
            "bagging_freq":      5,
            "min_child_samples": trial.suggest_int("min_child", 5, 50),
            "reg_alpha":         trial.suggest_float("alpha", 0.0, 5.0),
            "reg_lambda":        trial.suggest_float("lambda", 0.0, 5.0),
            "n_estimators":      2000,
            "verbose":           -1,
        }
        pruning_cb = LightGBMPruningCallback(trial, "rmse")
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(period=-1),
                pruning_cb,
            ],
        )
        return float(mean_squared_error(y_val, model.predict(X_val), squared=False))

    def optimise(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50,
    ) -> dict[str, Any]:
        split = int(len(X) * 0.8)
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )
        study.optimize(
            lambda t: self._lgb_objective(t, X[:split], y[:split], X[split:], y[split:]),
            n_trials=n_trials,
        )
        logger.info("LightGBM best RMSE: %.5f", study.best_value)
        best = study.best_params
        best.update({"objective": "regression", "metric": "rmse", "n_estimators": 2000, "verbose": -1})
        return best

    # --- Train ---------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y_return: pd.Series,
        y_direction: pd.Series,
    ) -> None:
        self.feature_names = list(X.columns)
        X_sc = self.scaler.fit_transform(X.values)
        split = int(len(X_sc) * 0.9)

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]

        self.reg_model = lgb.LGBMRegressor(**{**self.params, "objective": "regression"})
        self.reg_model.fit(
            X_sc[:split], y_return.values[:split],
            eval_set=[(X_sc[split:], y_return.values[split:])],
            callbacks=callbacks,
        )

        self.cls_model = lgb.LGBMClassifier(**{**self.params, "objective": "binary", "metric": "auc"})
        self.cls_model.fit(
            X_sc[:split], y_direction.values[:split],
            eval_set=[(X_sc[split:], y_direction.values[split:])],
            callbacks=callbacks,
        )

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        X_sc = self.scaler.transform(X.values)
        assert self.reg_model and self.cls_model
        return {
            "return":    self.reg_model.predict(X_sc),
            "direction": self.cls_model.predict_proba(X_sc)[:, 1],
        }

    def walk_forward_eval(
        self,
        X: pd.DataFrame,
        y_return: pd.Series,
        y_direction: pd.Series,
        n_folds: int = 5,
    ) -> pd.DataFrame:
        results = []
        for split in walk_forward_splits(len(X), n_folds=n_folds):
            self.fit(
                X.iloc[split.train_idx],
                y_return.iloc[split.train_idx],
                y_direction.iloc[split.train_idx],
            )
            preds = self.predict(X.iloc[split.val_idx])
            y_rv  = y_return.iloc[split.val_idx].values
            y_dv  = y_direction.iloc[split.val_idx].values

            results.append({
                "fold":    split.fold,
                "rmse":    mean_squared_error(y_rv, preds["return"], squared=False),
                "dir_acc": directional_accuracy(y_rv, preds["return"]),
                "sharpe":  sharpe_of_predictions(y_rv, preds["return"]),
                "auc":     roc_auc_score(y_dv, preds["direction"]),
            })
        return pd.DataFrame(results)

    def shap_importance(self, X: pd.DataFrame, n_samples: int = 500) -> pd.DataFrame:
        X_sc = self.scaler.transform(X.values[:n_samples])
        explainer = shap.TreeExplainer(self.reg_model)
        values = explainer.shap_values(X_sc)
        importance = np.abs(values).mean(axis=0)
        return pd.DataFrame({
            "feature":    self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.reg_model:
            self.reg_model.booster_.save_model(str(path / "lgb_reg.txt"))
        if self.cls_model:
            self.cls_model.booster_.save_model(str(path / "lgb_cls.txt"))
        joblib.dump(self.scaler, path / "scaler.pkl")
        joblib.dump(self.feature_names, path / "features.pkl")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        reg_booster = lgb.Booster(model_file=str(path / "lgb_reg.txt"))
        cls_booster = lgb.Booster(model_file=str(path / "lgb_cls.txt"))
        self.reg_model = lgb.LGBMRegressor()
        self.reg_model._Booster = reg_booster
        self.cls_model = lgb.LGBMClassifier()
        self.cls_model._Booster = cls_booster
        self.scaler        = joblib.load(path / "scaler.pkl")
        self.feature_names = joblib.load(path / "features.pkl")


# ---------------------------------------------------------------------------
# MLflow training run
# ---------------------------------------------------------------------------

def train_and_log(
    X: pd.DataFrame,
    y_return: pd.Series,
    y_direction: pd.Series,
    ticker: str,
    model_type: str = "xgboost",
    experiment_name: str = "quant-trees",
    n_optuna_trials: int = 30,
    n_folds: int = 5,
) -> str:
    """
    Full training pipeline: optimise → walk-forward eval → MLflow log → return run_id.
    """
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{model_type}_{ticker}") as run:
        trainer: XGBoostTrainer | LightGBMTrainer
        if model_type == "xgboost":
            trainer = XGBoostTrainer()
        else:
            trainer = LightGBMTrainer()

        mlflow.log_param("ticker", ticker)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("n_samples", len(X))
        mlflow.log_param("n_features", X.shape[1])

        # Optimise hyperparams on first 80% of data
        X_sc = StandardScaler().fit_transform(X.values)
        best_params = trainer.optimise(X_sc, y_return.values, n_trials=n_optuna_trials)
        mlflow.log_params(best_params)

        # Walk-forward evaluation
        scores = trainer.walk_forward_eval(X, y_return, y_direction, n_folds=n_folds)
        for col in ["rmse", "dir_acc", "sharpe", "auc"]:
            mlflow.log_metric(f"mean_{col}", float(scores[col].mean()))
            mlflow.log_metric(f"std_{col}",  float(scores[col].std()))

        logger.info(
            "%s %s | mean Sharpe=%.2f | mean DA=%.1f%%",
            model_type, ticker,
            scores["sharpe"].mean(), scores["dir_acc"].mean() * 100,
        )

        # Final fit on all data
        trainer.fit(X, y_return, y_direction)

        # Log SHAP importance
        shap_df = trainer.shap_importance(X)
        top10 = shap_df.head(10)
        for _, row in top10.iterrows():
            mlflow.log_metric(f"shap_{row['feature']}", float(row["importance"]))

        # Log model artefact
        save_dir = f"/tmp/{model_type}_{ticker}"
        trainer.save(save_dir)
        mlflow.log_artifacts(save_dir, artifact_path="model")

        return run.info.run_id
