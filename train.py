"""
Training and benchmarking pipeline.

Runs Gymnasium compliance check, evaluates OBI baseline,
trains SAC agent, and plots the learning curve.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from environment import LOBEnv


def check_gymnasium_compliance():
    from gymnasium.utils.env_checker import check_env

    print("=" * 60)
    print(" Gymnasium Compliance Check")
    print("=" * 60)

    env = LOBEnv("data/lob_features.parquet")
    check_env(env, warn=True)
    print("[OK] Gymnasium compliance: PASSED\n")

    print("ASCII Render Demo (3 random steps):")
    obs, _ = env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        env.render()
        print(f"  Reward: {reward:.6f} | Info: {info}\n")
    env.close()


def run_obi_baseline(env: LOBEnv, obi_threshold: float = 0.2) -> float:
    """Rule-based benchmark: buy when OBI > threshold, sell when < -threshold."""
    obs, _ = env.reset()
    total_reward, done = 0.0, False
    while not done:
        obi = obs[-4]
        if obi > obi_threshold:
            action = np.array([1.0, 1.0], dtype=np.float32)
        elif obi < -obi_threshold:
            action = np.array([-1.0, 1.0], dtype=np.float32)
        else:
            action = np.array([0.0, 0.0], dtype=np.float32)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
    return total_reward


def train_sac_agent(baseline_reward: float):
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import EvalCallback

    cfg = yaml.safe_load(open("config.yaml"))
    total_timesteps = cfg["training"]["total_timesteps"]
    n_envs = cfg["training"]["n_envs"]

    print("=" * 60)
    print(" SAC Agent Training")
    print("=" * 60)
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  OBI baseline to beat: {baseline_reward:.4f}")
    print()

    vec_env = make_vec_env(
        lambda: LOBEnv("data/lob_features.parquet"),
        n_envs=n_envs,
    )

    eval_env = LOBEnv("data/lob_features.parquet")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="data/",
        log_path="data/",
        eval_freq=max(total_timesteps // (n_envs * 20), 1000),
        n_eval_episodes=3,
        verbose=1,
    )

    model = SAC(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=50_000,
        learning_starts=1000,
        gamma=0.99,
        tau=0.005,
    )
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save("data/sac_lob_agent")
    print(f"\n[OK] Model saved to data/sac_lob_agent.zip")

    vec_env.close()
    eval_env.close()
    return model


def plot_learning_curve(baseline_reward: float):
    print("\n" + "=" * 60)
    print(" Learning Curve")
    print("=" * 60)

    results = np.load("data/evaluations.npz")
    timesteps = results["timesteps"]
    rewards   = results["results"].mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(timesteps, rewards, color="#1f77b4", lw=2, label="SAC agent", zorder=3)
    ax.fill_between(
        timesteps,
        results["results"].min(axis=1),
        results["results"].max(axis=1),
        alpha=0.15, color="#1f77b4",
    )
    ax.axhline(baseline_reward, color="#d62728", lw=1.5, linestyle="--",
               label=f"OBI baseline ({baseline_reward:.2f})", zorder=2)

    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Mean Episode Reward", fontsize=12)
    ax.set_title("SAC Agent vs OBI Baseline - LOB Execution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/learning_curve.png", dpi=150, bbox_inches="tight")
    print("[OK] Saved data/learning_curve.png")


if __name__ == "__main__":
    check_gymnasium_compliance()

    print("=" * 60)
    print(" OBI Baseline Evaluation")
    print("=" * 60)
    env = LOBEnv("data/lob_features.parquet")
    baseline = run_obi_baseline(env)
    env.close()
    print(f"  OBI baseline total reward: {baseline:.4f}\n")

    train_sac_agent(baseline)
    plot_learning_curve(baseline)

    print("\n" + "=" * 60)
    print(" All deliverables ready:")
    print("   - data/lob_features.parquet")
    print("   - data/features_plot.png")
    print("   - data/sac_lob_agent.zip")
    print("   - data/learning_curve.png")
    print("=" * 60)
