"""Data ingestion pipeline for L2 order book snapshots."""
import os
import yaml
import pandas as pd

cfg = yaml.safe_load(open("config.yaml"))["data"]

try:
    from pulse import PulseClient
    from utils import parse_lob_snapshot

    api_key = os.environ["PULSE_API_KEY"]
    client = PulseClient(api_key=api_key)

    print(f"Fetching {cfg['symbol']} from Pulse API ...")
    raw = client.get_orderbook(
        symbol=cfg["symbol"],
        start=cfg["start"],
        end=cfg["end"],
        levels=cfg["levels"],
        resolution=cfg["resolution"],
    )
    df = parse_lob_snapshot(raw, levels=cfg["levels"])
    print(f"Pulse API: ingested {len(df):,} ticks")

except Exception as e:
    print(f"Pulse API unavailable ({type(e).__name__}: {e})")
    print("Falling back to synthetic data generator ...")
    from utils import generate_synthetic_lob

    df = generate_synthetic_lob(
        n_ticks=10_000,
        levels=cfg["levels"],
        initial_price=185.50,
        tick_size=0.01,
        seed=42,
    )
    print(f"Synthetic: generated {len(df):,} ticks")

os.makedirs("data", exist_ok=True)
df.to_parquet("data/lob_slice.parquet")
print(f"Saved to data/lob_slice.parquet")
print(f"  Spread mean: {df['spread'].mean():.5f}")
print(f"  Mid-price range: [{df['mid_price'].min():.2f}, {df['mid_price'].max():.2f}]")
print(f"  Columns: {list(df.columns)}")
