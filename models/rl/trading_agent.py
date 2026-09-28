"""
models/rl/trading_agent.py
Reinforcement learning trading agent using Stable-Baselines3 PPO.
The environment models realistic trading with transaction costs,
position limits, and portfolio-level reward shaping.
"""

from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from config import get_settings

settings = get_settings()
trade_cfg = settings.trading
rl_cfg = settings.model


# ─────────────────────────────────────────────────────────────────────────────
# Trading Environment
# ─────────────────────────────────────────────────────────────────────────────

class TradingEnv(gym.Env):
    """
    Multi-asset trading environment.

    Observation:
        Flattened feature matrix for all tickers + portfolio state.

    Action space:
        Continuous actions in [-1, 1] per ticker.
        Positive → long, negative → short, 0 → hold.
        Actions are portfolio weights (normalised to sum to 1 in magnitude).

    Reward:
        Risk-adjusted return (Sortino-like) minus transaction costs.
        Penalises excessive drawdown and position concentration.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: dict[str, np.ndarray],   # ticker → (T, F) feature arrays
        prices: dict[str, np.ndarray],      # ticker → price series
        lookback: int = 20,
        initial_equity: float = 1_000_000.0,
        commission_pct: float = trade_cfg.commission_pct,
        slippage_pct: float = trade_cfg.slippage_pct,
        max_position_pct: float = trade_cfg.max_position_pct,
        reward_scaling: float = 100.0,
    ):
        super().__init__()
        self.tickers = sorted(features.keys())
        self.n_tickers = len(self.tickers)
        self.features = {t: features[t] for t in self.tickers}
        self.prices = {t: prices[t] for t in self.tickers}
        self.lookback = lookback
        self.initial_equity = initial_equity
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.max_position_pct = max_position_pct
        self.reward_scaling = reward_scaling

        # Infer feature dimension from first ticker
        first_feat = next(iter(features.values()))
        self.n_features = first_feat.shape[1]
        self.T = first_feat.shape[0]

        # Observation: lookback × features per ticker + portfolio state
        portfolio_state_dim = self.n_tickers + 3   # positions + equity + drawdown + cash%
        obs_dim = (self.n_tickers * self.n_features * lookback) + portfolio_state_dim

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Continuous portfolio weights per ticker
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_tickers,), dtype=np.float32
        )

        self._portfolio: dict[str, float] = {}   # ticker → shares
        self._equity = initial_equity
        self._cash = initial_equity
        self._peak_equity = initial_equity
        self._t = lookback
        self._returns: list[float] = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._portfolio = {t: 0.0 for t in self.tickers}
        self._equity = self.initial_equity
        self._cash = self.initial_equity
        self._peak_equity = self.initial_equity
        self._t = self.lookback
        self._returns = []
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._t >= self.T:
            return self._get_obs(), 0.0, True, False, {}

        prev_equity = self._equity

        # ── Clip actions and normalise to portfolio weights ───────────────
        action = np.clip(action, -1.0, 1.0)
        total_exposure = np.sum(np.abs(action))
        if total_exposure > 1.0:
            action = action / total_exposure

        # ── Execute trades ────────────────────────────────────────────────
        total_commission = 0.0
        for i, ticker in enumerate(self.tickers):
            target_weight = float(action[i])
            current_price = self.prices[ticker][self._t]
            if current_price == 0:
                continue

            target_value = self._equity * target_weight * self.max_position_pct
            current_value = self._portfolio.get(ticker, 0.0) * current_price

            trade_value = target_value - current_value
            if abs(trade_value) < 100:   # ignore tiny rebalances
                continue

            # Apply slippage
            if trade_value > 0:
                exec_price = current_price * (1 + self.slippage_pct)
            else:
                exec_price = current_price * (1 - self.slippage_pct)

            shares_delta = trade_value / exec_price
            commission = abs(trade_value) * self.commission_pct

            # Check cash constraint
            cost = abs(trade_value) + commission
            if trade_value > 0 and cost > self._cash:
                shares_delta *= self._cash / cost
                cost = self._cash

            self._portfolio[ticker] = self._portfolio.get(ticker, 0.0) + shares_delta
            self._cash -= trade_value + commission
            total_commission += commission

        # ── Advance time ──────────────────────────────────────────────────
        self._t += 1

        # ── Update equity ─────────────────────────────────────────────────
        pos_value = sum(
            self._portfolio.get(t, 0.0) * self.prices[t][self._t]
            for t in self.tickers
        )
        self._equity = self._cash + pos_value
        self._peak_equity = max(self._peak_equity, self._equity)

        # ── Compute reward ────────────────────────────────────────────────
        daily_return = (self._equity - prev_equity) / prev_equity
        self._returns.append(daily_return)

        reward = self._compute_reward(daily_return, total_commission)

        done = self._t >= self.T - 1
        truncated = False

        info = {
            "equity": self._equity,
            "daily_return": daily_return,
            "drawdown": (self._peak_equity - self._equity) / self._peak_equity,
            "commission": total_commission,
        }
        return self._get_obs(), float(reward), done, truncated, info

    def _compute_reward(self, daily_return: float, commission: float) -> float:
        """
        Reward = Sortino-like risk-adjusted return - commission penalty.
        Penalise drawdown non-linearly (convex).
        """
        # Risk-adjusted return
        if len(self._returns) >= 20:
            downside = [r for r in self._returns[-60:] if r < 0]
            downside_std = np.std(downside) if downside else 1e-6
            sortino_component = daily_return / (downside_std + 1e-8)
        else:
            sortino_component = daily_return

        # Drawdown penalty (convex in drawdown magnitude)
        dd = (self._peak_equity - self._equity) / self._peak_equity
        dd_penalty = -5.0 * (dd ** 2) if dd > 0.05 else 0.0

        # Transaction cost penalty
        cost_penalty = -commission / self._equity * 100

        return (sortino_component * self.reward_scaling + dd_penalty + cost_penalty)

    def _get_obs(self) -> np.ndarray:
        """Build observation vector: feature lookback + portfolio state."""
        feature_parts = []
        for ticker in self.tickers:
            feat_matrix = self.features[ticker]
            start = max(0, self._t - self.lookback)
            window = feat_matrix[start: self._t]
            # Pad if shorter than lookback
            if len(window) < self.lookback:
                pad = np.zeros((self.lookback - len(window), self.n_features))
                window = np.vstack([pad, window])
            feature_parts.append(window.flatten())

        # Portfolio state
        portfolio_state = np.array([
            self._portfolio.get(t, 0.0) * self.prices[t][self._t] / self._equity
            for t in self.tickers
        ] + [
            self._equity / self.initial_equity,        # normalised equity
            (self._peak_equity - self._equity) / self._peak_equity,  # drawdown
            self._cash / self._equity,                  # cash ratio
        ], dtype=np.float32)

        obs = np.concatenate(feature_parts + [portfolio_state]).astype(np.float32)
        obs = np.clip(obs, -10.0, 10.0)   # safety clip
        return obs

    def render(self):
        dd = (self._peak_equity - self._equity) / self._peak_equity
        print(f"t={self._t} | equity=${self._equity:,.0f} | dd={dd:.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class SharpeCallback(BaseCallback):
    """
    Custom callback that logs the Sharpe ratio at each evaluation.
    Stops training early if we achieve a target Sharpe.
    """

    def __init__(self, target_sharpe: float = 2.0, verbose: int = 0):
        super().__init__(verbose)
        self.target_sharpe = target_sharpe
        self._episode_returns: list[float] = []

    def _on_step(self) -> bool:
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "daily_return" in info:
                    self._episode_returns.append(info["daily_return"])

        if len(self._episode_returns) >= 252:
            rets = np.array(self._episode_returns[-252:])
            sharpe = (rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252)
            self.logger.record("custom/sharpe_252", sharpe)
            if sharpe >= self.target_sharpe:
                if self.verbose:
                    print(f"Target Sharpe {self.target_sharpe} reached. Stopping.")
                return False
        return True


class RLTradingTrainer:
    """Trains a PPO (or SAC) agent on the trading environment."""

    def __init__(
        self,
        features: dict[str, np.ndarray],
        prices: dict[str, np.ndarray],
        algorithm: str = "PPO",
        n_envs: int = 4,
    ):
        self.features = features
        self.prices = prices
        self.algorithm = algorithm
        self.n_envs = n_envs
        self.model: Optional[PPO | SAC] = None

    def build_env(self, normalise: bool = True) -> VecNormalize:
        def make():
            env = TradingEnv(self.features, self.prices)
            return Monitor(env)

        vec_env = DummyVecEnv([make] * self.n_envs)
        if normalise:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        return vec_env

    def train(
        self,
        total_timesteps: int = 1_000_000,
        save_path: str = "models/rl/checkpoints",
    ) -> PPO | SAC:
        env = self.build_env()

        policy_kwargs = {
            "net_arch": [dict(pi=[256, 256, 128], vf=[256, 256, 128])],
            "activation_fn": __import__("torch").nn.Tanh,
        }

        callbacks = [
            CheckpointCallback(save_freq=50_000, save_path=save_path),
            SharpeCallback(target_sharpe=2.0, verbose=1),
        ]

        if self.algorithm == "PPO":
            self.model = PPO(
                "MlpPolicy",
                env,
                learning_rate=rl_cfg.rl_learning_rate,
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="logs/rl/",
            )
        else:   # SAC
            self.model = SAC(
                "MlpPolicy",
                env,
                learning_rate=rl_cfg.rl_learning_rate,
                buffer_size=1_000_000,
                batch_size=256,
                gamma=0.99,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log="logs/rl/",
            )

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        return self.model

    def evaluate(
        self,
        n_episodes: int = 5,
    ) -> dict:
        """Run n evaluation episodes and return aggregate metrics."""
        if self.model is None:
            raise RuntimeError("Model not trained yet")
        env = TradingEnv(self.features, self.prices)
        all_returns = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_returns = []
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                if "daily_return" in info:
                    ep_returns.append(info["daily_return"])
            all_returns.extend(ep_returns)

        rets = np.array(all_returns)
        return {
            "total_return_pct": round((np.cumprod(1 + rets)[-1] - 1) * 100, 2),
            "sharpe": round((rets.mean() / (rets.std() + 1e-8)) * np.sqrt(252), 3),
            "max_drawdown_pct": round(
                (1 - np.min(np.cumprod(1 + rets) / np.maximum.accumulate(np.cumprod(1 + rets)))) * 100, 2
            ),
        }
