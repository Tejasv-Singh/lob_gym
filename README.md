# Limit Order Book (LOB) RL Gym

A 4-phase reinforcement learning pipeline and custom Gymnasium environment designed for high-frequency execution inside a Limit Order Book. This project bridges the gap between traditional discrete-action toy models and realistic, continuous microstructure trading by leveraging continuous action spaces (Soft Actor-Critic) and mark-to-market adverse selection rewards.

## Project Structure

The project is broken into four distinct phases:

1. **`ingest.py` (Phase 1: Data Ingestion)**
   Fetches L2 order book snapshots. Currently features a fallback synthetic microstructure generator (Ornstein-Uhlenbeck mid-price, clustered spread regimes, and correlated volume bursts) designed to mimic institutional flow.
   
2. **`features.py` (Phase 2: Microstructure Feature Engineering)**
   Computes advanced order book signals (building on the framework defined by **Cont, Kukanov & Stoikov, 2014**):
   - **OBI**: Standard Order Book Imbalance.
   - **OFI**: Order Flow Imbalance.
   - **Weighted OBI**: Depth-decayed imbalance using harmonic weights.

3. **`environment.py` (Phase 3: Gymnasium LOBEnv)**
   A fully-compliant custom `gymnasium` environment. 
   - **Action Space**: Continuous 2D array `[direction, size_fraction]`.
   - **Reward Formulation**: Uses a $k$-step lookahead mark-to-market PnL to directly penalize adverse selection (getting "picked off").
   - **Renderer**: Built-in ASCII depth ladder for terminal-based visualization.

4. **`train.py` (Phase 4: SAC Agent Training)**
   Implements a stable-baselines3 Soft Actor-Critic (SAC) agent. Benchmarks the RL policy against a deterministic, rule-based OBI baseline model.

## Installation

Ensure you have Python 3.9+ installed.

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install pandas numpy gymnasium stable-baselines3 pyyaml matplotlib pyarrow fastparquet
```

## Usage

You can run the pipeline sequentially:

```bash
# 1. Generate the synthetic orderbook data (10k ticks)
python ingest.py

# 2. Compute microstructure features and correlation statistics
python features.py

# 3. Train the SAC agent and plot the learning curve
python train.py
```

## Deliverables & Analysis

All generated artifacts are saved to the `data/` directory:
- `features_plot.png`: Visualization of the OBI, OFI, and Weighted OBI signals against the mid-price.
- `learning_curve.png`: The training trajectory of the SAC agent vs the OBI threshold baseline.
- `sac_lob_agent.zip`: The final serialized stable-baselines3 model.

---
*Built to bridge continuous RL algorithms with realistic microstructure mechanics.*
