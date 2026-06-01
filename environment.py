"""
Gymnasium environment for limit order book execution.

Observation: normalised bid/ask prices and sizes for N levels,
             plus OBI, OFI, Weighted OBI, and a market_pressure placeholder.
Action:      continuous [direction(-1..1), size_fraction(0..1)].
Reward:      mark-to-market PnL k steps ahead (adverse selection proxy).
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple
import yaml


class LOBEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, parquet_path: str, config_path: str = "config.yaml"):
        super().__init__()
        cfg = yaml.safe_load(open(config_path))
        env_cfg = cfg["env"]

        self.df              = pd.read_parquet(parquet_path)
        self.n_levels        = cfg["data"]["levels"]
        self.max_trade_size  = env_cfg["trade_size"]
        self.impact_penalty  = env_cfg["impact_penalty"]
        self.k               = env_cfg["mark_to_market_k"]
        self.hold_penalty    = env_cfg["hold_penalty"]

        self._current_step    = 0
        self._position        = 0.0
        self._entry_price     = 0.0
        self._market_pressure = 0.0

        obs_dim = self.n_levels * 4 + 4
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._current_step  = 0
        self._position      = 0.0
        self._entry_price   = 0.0
        return self._get_observation(), {}

    def step(self, action: np.ndarray):
        direction  = float(np.clip(action[0], -1.0, 1.0))
        size_frac  = float(np.clip(action[1],  0.0, 1.0))
        qty        = int(size_frac * self.max_trade_size)

        row    = self.df.iloc[self._current_step]
        mid    = row["mid_price"]
        reward = 0.0
        info   = {
            "mid_price": mid,
            "position": self._position,
            "step": self._current_step,
        }

        if qty > 0 and abs(direction) > 0.1:
            if direction > 0:
                exec_price, filled = self._walk_book(row, "ask", qty)
                self._position    += filled
                self._entry_price  = exec_price
                reward             = self._mark_to_market(exec_price, "buy", filled)
                reward            -= self.impact_penalty * filled ** 2
                info.update({"exec_price": exec_price, "filled": filled, "side": "buy"})
            else:
                exec_price, filled = self._walk_book(row, "bid", qty)
                self._position    -= filled
                reward             = self._mark_to_market(exec_price, "sell", filled)
                reward            -= self.impact_penalty * filled ** 2
                info.update({"exec_price": exec_price, "filled": filled, "side": "sell"})
        else:
            reward = -self.hold_penalty * abs(self._position)

        self._current_step += 1
        terminated = self._current_step >= len(self.df) - self.k - 1
        return self._get_observation(), reward, terminated, False, info

    def render(self, mode="ansi"):
        """ASCII depth ladder."""
        row = self.df.iloc[self._current_step]
        mid = row["mid_price"]
        bar_scale = 0.5

        lines = []
        lines.append(f"\n{'-' * 44}")
        lines.append(f"  Step {self._current_step:>6} | Mid: {mid:.4f} | Pos: {self._position:+.0f}")
        lines.append(f"{'-' * 44}")

        for i in range(self.n_levels, 0, -1):
            ask_p = row[f"ask_price_{i}"]
            ask_s = int(row[f"ask_size_{i}"])
            bar   = "#" * min(int(ask_s * bar_scale), 20)
            lines.append(f"  ASK L{i}  {ask_p:.4f}  {bar:<20}  {ask_s:>6}")

        lines.append(f"  {'.' * 40}")

        for i in range(1, self.n_levels + 1):
            bid_p = row[f"bid_price_{i}"]
            bid_s = int(row[f"bid_size_{i}"])
            bar   = "#" * min(int(bid_s * bar_scale), 20)
            lines.append(f"  BID L{i}  {bid_p:.4f}  {bar:<20}  {bid_s:>6}")

        lines.append(f"  OBI: {row['obi']:+.3f}  OFI: {row['ofi']:+.1f}")
        lines.append(f"{'-' * 44}\n")

        output = "\n".join(lines)
        print(output)
        return output

    def _get_observation(self) -> np.ndarray:
        row = self.df.iloc[min(self._current_step, len(self.df) - 1)]
        mid = row["mid_price"]
        features = []
        for i in range(1, self.n_levels + 1):
            features.append((row[f"bid_price_{i}"] - mid) / mid)
            features.append(row[f"bid_size_{i}"])
            features.append((row[f"ask_price_{i}"] - mid) / mid)
            features.append(row[f"ask_size_{i}"])
        features.append(float(row["obi"]))
        features.append(float(row["ofi"]))
        features.append(float(row["weighted_obi"]))
        features.append(self._market_pressure)
        return np.array(features, dtype=np.float32)

    def _walk_book(self, row: pd.Series, side: str, qty: int) -> Tuple[float, int]:
        """Walk through order book levels to fill an order, return VWAP and filled qty."""
        remaining, total_cost, filled = qty, 0.0, 0
        for i in range(1, self.n_levels + 1):
            if remaining <= 0:
                break
            price = row[f"{side}_price_{i}"]
            avail = int(row[f"{side}_size_{i}"])
            take  = min(remaining, avail)
            total_cost += price * take
            filled     += take
            remaining  -= take
        vwap = total_cost / filled if filled > 0 else row[f"{side}_price_1"]
        return vwap, filled

    def _mark_to_market(self, exec_price: float, side: str, filled: int) -> float:
        """Mark position to mid-price k steps ahead (adverse selection proxy)."""
        future_idx = min(self._current_step + self.k, len(self.df) - 1)
        future_mid = self.df.iloc[future_idx]["mid_price"]
        if side == "buy":
            return (future_mid - exec_price) * filled
        else:
            return (exec_price - future_mid) * filled
