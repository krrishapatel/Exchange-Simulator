"""Training script for PPO agent on the TradingEnv.

Requires stable-baselines3: pip install stable-baselines3
"""

from __future__ import annotations

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on TradingEnv")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps (default: 100000)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/ppo_trader",
        help="Path to save the trained model (default: models/ppo_trader)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/ppo_trading",
        help="Tensorboard log directory (default: logs/ppo_trading)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError:
        print(
            "ERROR: stable-baselines3 is required for training.\n"
            "Install with: pip install 'stable-baselines3>=2.0'",
            file=sys.stderr,
        )
        sys.exit(1)

    from rl import TradingEnv

    # Create output directories
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print(f"Creating TradingEnv with seed={args.seed}")
    env = TradingEnv(seed=args.seed)

    print(f"Training PPO for {args.total_timesteps} timesteps...")
    print(f"  Model will be saved to: {args.model_path}.zip")
    print(f"  Tensorboard logs: {args.log_dir}")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=args.log_dir,
        seed=args.seed,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    model.learn(total_timesteps=args.total_timesteps)

    model.save(args.model_path)
    print(f"\nModel saved to {args.model_path}.zip")

    # Quick evaluation
    print("\nRunning evaluation episode...")
    obs, info = env.reset(seed=args.seed + 1)
    total_reward = 0.0
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        if terminated or truncated:
            break

    print(f"Evaluation episode reward: {total_reward:.2f}")
    print(f"Final inventory: {info['inventory']}")
    print(f"Final PnL: {info['pnl']:.2f}")


if __name__ == "__main__":
    main()
