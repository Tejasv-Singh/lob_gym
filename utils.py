"""Utility functions for LOB data parsing and synthetic data generation."""
import numpy as np
import pandas as pd


def parse_lob_snapshot(raw: list, levels: int = 5) -> pd.DataFrame:
    """Parse raw Pulse API order book snapshots into a flat DataFrame.

    Returns DataFrame with columns:
        bid_price_1..N, bid_size_1..N, ask_price_1..N, ask_size_1..N,
        mid_price, spread
    """
    records = []
    for snap in raw:
        row = {"timestamp": pd.Timestamp(snap["timestamp"])}
        bids = snap.get("bids", [])
        asks = snap.get("asks", [])

        for i in range(levels):
            if i < len(bids):
                row[f"bid_price_{i+1}"] = float(bids[i][0])
                row[f"bid_size_{i+1}"]  = float(bids[i][1])
            else:
                row[f"bid_price_{i+1}"] = np.nan
                row[f"bid_size_{i+1}"]  = 0.0

            if i < len(asks):
                row[f"ask_price_{i+1}"] = float(asks[i][0])
                row[f"ask_size_{i+1}"]  = float(asks[i][1])
            else:
                row[f"ask_price_{i+1}"] = np.nan
                row[f"ask_size_{i+1}"]  = 0.0

        row["mid_price"] = (row["bid_price_1"] + row["ask_price_1"]) / 2
        row["spread"]    = row["ask_price_1"] - row["bid_price_1"]
        records.append(row)

    df = pd.DataFrame(records)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df


def generate_synthetic_lob(
    n_ticks: int = 10_000,
    levels: int = 5,
    initial_price: float = 185.50,
    tick_size: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic L2 order book data with realistic microstructure.

    Features: OU mid-price, clustered spread regimes, exponential depth
    decay, and correlated volume bursts simulating informed flow.
    """
    rng = np.random.default_rng(seed)

    mu, theta, sigma = initial_price, 0.005, 0.02
    prices = np.empty(n_ticks)
    prices[0] = initial_price
    for t in range(1, n_ticks):
        dp = theta * (mu - prices[t - 1]) + sigma * rng.standard_normal()
        prices[t] = prices[t - 1] + dp

    spread_regime = np.ones(n_ticks)
    in_wide = False
    for t in range(n_ticks):
        if rng.random() < 0.002:
            in_wide = not in_wide
        spread_regime[t] = 3 if in_wide else 1

    spreads = spread_regime * tick_size + rng.exponential(0.5 * tick_size, n_ticks)
    spreads = np.maximum(spreads, tick_size)

    timestamps = pd.date_range(
        start="2024-01-15 09:30:00", periods=n_ticks, freq="ms", tz="UTC",
    )

    records = {"mid_price": prices, "spread": spreads}
    best_bid = prices - spreads / 2
    best_ask = prices + spreads / 2

    base_volume, decay_rate = 200.0, 0.6

    vol_burst = np.ones(n_ticks)
    for t in range(n_ticks):
        if rng.random() < 0.01:
            burst_len = rng.integers(10, 100)
            vol_burst[t: t + burst_len] = rng.uniform(2.0, 5.0)

    for i in range(1, levels + 1):
        depth = decay_rate ** (i - 1)
        records[f"bid_price_{i}"] = best_bid - (i - 1) * tick_size
        records[f"bid_size_{i}"]  = np.maximum(
            (base_volume * depth * vol_burst * rng.lognormal(0, 0.4, n_ticks)).astype(int), 1
        ).astype(float)
        records[f"ask_price_{i}"] = best_ask + (i - 1) * tick_size
        records[f"ask_size_{i}"]  = np.maximum(
            (base_volume * depth * vol_burst * rng.lognormal(0, 0.4, n_ticks)).astype(int), 1
        ).astype(float)

    df = pd.DataFrame(records, index=timestamps)
    df.index.name = "timestamp"
    price_cols = [c for c in df.columns if "price" in c]
    df[price_cols] = df[price_cols].round(2)
    return df
