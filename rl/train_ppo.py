"""Train PPO on TradingEnv, then measure it against the heuristic baselines.

Requires stable-baselines3: pip install stable-baselines3

Two things this writes that the previous version did not:

  * a learning curve. Episodes are logged through Monitor to
    ``<log-dir>/monitor.csv``, so training progress is a file on disk rather than
    scrollback. Without it there is no way to say whether a run learned.
  * an evaluation on seeds the agent never trained on, against the policies in
    baselines.py. A reward number by itself is unreadable, since the reward
    includes an inventory penalty and is measured in ticks. Beating
    ``inventory_aware`` is the claim worth making.

Both are reported on marked equity, not cash. See TradingEnv._marked_equity.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys


def evaluate_policy_fn(env, act, seeds):
    """Run a callable policy over the given seeds, returning per-episode stats.

    ``act`` takes (obs, info) and returns an action, which is what the baselines
    expose and what a wrapped model can be made to look like.
    """
    rows = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        inventories = []
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(act(obs, info))
            total_reward += reward
            inventories.append(abs(info["inventory"]))
        rows.append(
            {
                "equity": info["equity"],
                "reward": total_reward,
                "mean_abs_inv": statistics.mean(inventories),
                "max_abs_inv": max(inventories),
            }
        )
    return rows


def summarize(name, rows):
    """One line per policy: mean equity with a standard error, and position size."""
    equity = [r["equity"] for r in rows]
    mean = statistics.mean(equity)
    stderr = statistics.stdev(equity) / len(equity) ** 0.5 if len(equity) > 1 else 0.0
    return (
        f"{name:<20} {mean:>9.1f} +/- {stderr:<6.1f}  "
        f"{statistics.mean(r['mean_abs_inv'] for r in rows):>9.1f}  "
        f"{statistics.mean(r['max_abs_inv'] for r in rows):>8.1f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on TradingEnv")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200_000,
        help="Total training timesteps (default: 200000)",
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
        help="Directory for monitor.csv and tensorboard logs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Held-out episodes per policy in the final comparison (default: 20)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable observation and reward normalization (on by default)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--inventory-penalty",
        type=float,
        default=None,
        help="Override the env's per-step penalty on abs(inventory)",
    )
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError:
        print(
            "ERROR: stable-baselines3 is required for training.\n"
            "Install with: pip install 'stable-baselines3>=2.0'",
            file=sys.stderr,
        )
        sys.exit(1)

    from rl import TradingEnv, all_baselines

    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Monitor writes one row per episode to monitor.csv. That file is the
    # learning curve. Monitor sits under VecNormalize so its rewards are the real
    # ones in ticks rather than the normalized ones.
    env_kwargs = {"seed": args.seed}
    if args.inventory_penalty is not None:
        env_kwargs["inventory_penalty"] = args.inventory_penalty

    print(f"Creating TradingEnv with seed={args.seed}")
    env = Monitor(TradingEnv(**env_kwargs), os.path.join(args.log_dir, "monitor"))

    # Reward normalization matters more here than it usually does. The reward is
    # the change in marked equity in ticks, so a step is worth about 0.02 when
    # the agent is earning the spread and about 0.3 * inventory when the mid
    # moves under a position. At 10 units held that is a signal two orders of
    # magnitude below the noise carrying it, and an unnormalized value function
    # spends its capacity on the noise. VecNormalize rescales both by a running
    # estimate, which is the difference between a value function that explains
    # something and one that does not.
    vec_env = DummyVecEnv([lambda: env])
    normalize = not args.no_normalize
    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Tensorboard is optional. SB3 raises if tensorboard_log is set without it
    # installed, and it is not in requirements.txt, so asking for it by default
    # made this script fail before it trained anything. monitor.csv is the curve
    # either way.
    try:
        import tensorboard  # noqa: F401

        tensorboard_log = args.log_dir
    except ImportError:
        tensorboard_log = None

    print(f"Training PPO for {args.total_timesteps} timesteps...")
    print(f"  Model will be saved to: {args.model_path}.zip")
    print(f"  Episode log: {os.path.join(args.log_dir, 'monitor.csv')}")
    if tensorboard_log is None:
        print("  Tensorboard not installed, skipping tensorboard logs")

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        seed=args.seed,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    model.learn(total_timesteps=args.total_timesteps)

    model.save(args.model_path)
    print(f"\nModel saved to {args.model_path}.zip")
    if normalize:
        # The running mean and variance are part of the policy. A checkpoint
        # loaded without them sees inputs on a different scale than it trained
        # on and behaves like an untrained network.
        vec_env.save(f"{args.model_path}_vecnormalize.pkl")
        vec_env.training = False
        print(f"Normalizer saved to {args.model_path}_vecnormalize.pkl")

    # Evaluate on seeds well clear of the training seed, so this is not a report
    # on episodes the agent has already seen.
    eval_seeds = [100_000 + i for i in range(args.eval_episodes)]
    eval_env = TradingEnv(seed=args.seed + 1)

    print(f"\nHeld-out evaluation, {len(eval_seeds)} episodes per policy")
    print(f"{'policy':<20} {'equity':>18}  {'mean|inv|':>9}  {'max|inv|':>8}")

    def model_act(obs, info):
        if normalize:
            obs = vec_env.normalize_obs(obs)
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    trained_rows = evaluate_policy_fn(eval_env, model_act, eval_seeds)
    print(summarize("PPO (trained)", trained_rows))

    for policy in all_baselines():
        policy.reset()
        rows = evaluate_policy_fn(
            eval_env, lambda o, i, p=policy: p.act(o, i), eval_seeds
        )
        print(summarize(policy.name, rows))

    # The action mix says what the policy actually does. A single repeated action
    # is the failure mode to watch for: an earlier run on the pre-fix env came out
    # 100% sell_limit, which is a degenerate policy and not a strategy.
    counts = [0] * 5
    obs, info = eval_env.reset(seed=eval_seeds[0])
    terminated = truncated = False
    while not (terminated or truncated):
        action = model_act(obs, info)
        counts[action] += 1
        obs, _, terminated, truncated, info = eval_env.step(action)
    names = ["hold", "buy_limit", "buy_market", "sell_limit", "sell_market"]
    total = sum(counts)
    print("\nAction mix on one held-out episode:")
    for name, count in zip(names, counts):
        print(f"  {name:<12} {count / total:>6.1%}")


if __name__ == "__main__":
    main()
