"""Data ingestion pipeline for L2 order book snapshots."""
import os
import yaml
import pandas as pd

cfg = yaml.safe_load(open("config.yaml"))["data"]

try:
    from simudyne import PulseABM

    api_key = os.environ["PULSE_API_KEY"]
    client = PulseABM(api_key=api_key)

    print(f"Fetching cached simulation for {cfg['symbol']} from Pulse API ...")
    cached = client.simulation.list_cached(symbol=cfg['symbol'], date=cfg.get('cal_date'))
    
    # Grab the first available cached simulation for this symbol
    sim_id = cached["simulations"][0]["example_sim_id"]
    polars_df = client.simulation.get_sim_data(sim_id)
    
    # Pulse API returns polars.DataFrame; convert to Pandas for the Gym environment
    df = polars_df.to_pandas()
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

# Drop rows where the order book is empty (e.g. pre-market auction phase)
df.dropna(subset=["ask_price_1", "bid_price_1"], inplace=True)

# Fill sparse deeper levels with 0 so downstream math doesn't result in NaNs
df.fillna(0, inplace=True)

if "spread" not in df.columns:
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
if "mid_price" not in df.columns:
    df["mid_price"] = (df["ask_price_1"] + df["bid_price_1"]) / 2

df.to_parquet("data/lob_slice.parquet")
print(f"Saved to data/lob_slice.parquet")
print(f"  Spread mean: {df['spread'].mean():.5f}")
print(f"  Mid-price range: [{df['mid_price'].min():.2f}, {df['mid_price'].max():.2f}]")
print(f"  Columns: {list(df.columns)}")
