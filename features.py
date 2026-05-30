"""
Feature engineering for L2 order book data.

Signals computed:
  1. OBI  -- Order Book Imbalance (snapshot bid vs ask volume ratio)
  2. OFI  -- Order Flow Imbalance (Cont, Kukanov & Stoikov, 2014)
  3. Weighted OBI -- depth-decayed imbalance with harmonic weights
"""
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

cfg = yaml.safe_load(open("config.yaml"))
levels = cfg["data"]["levels"]

df = pd.read_parquet("data/lob_slice.parquet")
print(f"Loaded {len(df):,} ticks from data/lob_slice.parquet")


def compute_obi(df: pd.DataFrame, levels: int = 5) -> pd.Series:
    bid_vol = sum(df[f"bid_size_{i}"] for i in range(1, levels + 1))
    ask_vol = sum(df[f"ask_size_{i}"] for i in range(1, levels + 1))
    return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)

df["obi"] = compute_obi(df, levels)

df["ofi"] = (
    df["bid_size_1"].diff().clip(lower=0)
    - df["ask_size_1"].diff().clip(lower=0)
).fillna(0)

weights = np.array([1 / i for i in range(1, levels + 1)])
df["weighted_obi"] = (
    sum(
        weights[i - 1] * (df[f"bid_size_{i}"] - df[f"ask_size_{i}"])
        / (df[f"bid_size_{i}"] + df[f"ask_size_{i}"] + 1e-9)
        for i in range(1, levels + 1)
    )
    / weights.sum()
)


def signal_analysis(df: pd.DataFrame, lookahead_ms: int = 500):
    """Correlate features against forward returns and print conditional means."""
    df = df.copy()
    df["fwd_return"] = df["mid_price"].pct_change(lookahead_ms).shift(-lookahead_ms)

    print("")
    print("=" * 60)
    print(f" Signal -> {lookahead_ms}ms forward return correlations")
    print("=" * 60)
    for feat in ["obi", "ofi", "weighted_obi"]:
        corr = df[[feat, "fwd_return"]].dropna().corr().iloc[0, 1]
        print(f"  {feat:>15} -> {corr:+.4f}")

    try:
        df["ofi_bin"] = pd.qcut(df["ofi"], q=5, duplicates="drop")
    except ValueError:
        df["ofi_bin"] = pd.qcut(df["ofi"].rank(method="first"), q=5)
    print(f"\n  Conditional avg forward return by OFI quintile:")
    cond_means = df.groupby("ofi_bin", observed=True)["fwd_return"].mean()
    quintile_names = [f"Q{i+1}" for i in range(len(cond_means))]
    for i, (label, val) in enumerate(cond_means.items()):
        name = quintile_names[i]
        print(f"    {name:>14}: {val:+.6f}")

signal_analysis(df)


def plot_features(df: pd.DataFrame):
    """Four-panel chart: mid-price + OBI + OFI + Weighted OBI."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(df.index, df["mid_price"], lw=0.8, color="#1f77b4")
    axes[0].set_ylabel("Mid price", fontsize=11)
    axes[0].set_title("LOB Microstructure Features vs Mid-Price", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    features = [
        ("obi",          "#2ca02c", "OBI"),
        ("ofi",          "#d62728", "OFI (Cont et al.)"),
        ("weighted_obi", "#9467bd", "Weighted OBI"),
    ]
    for ax, (col, color, label) in zip(axes[1:], features):
        ax.fill_between(
            df.index, df[col], 0,
            where=(df[col] > 0), color=color, alpha=0.4, label=f"{label} > 0"
        )
        ax.fill_between(
            df.index, df[col], 0,
            where=(df[col] < 0), color=color, alpha=0.15, label=f"{label} < 0"
        )
        ax.axhline(0, color="black", lw=0.4)
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("data/features_plot.png", dpi=150, bbox_inches="tight")
    print("\nSaved data/features_plot.png")

plot_features(df)

df.to_parquet("data/lob_features.parquet")
print(f"Saved data/lob_features.parquet ({len(df):,} rows, {len(df.columns)} cols)")
