"""Evaluation script for trained RL trading agents.

Runs a trained model against baseline opponents and reports:
  - Total PnL
  - Sharpe ratio
  - Max drawdown
  - Inventory utilization

Usage:
    python rl/evaluate.py --model models/ppo_trader.zip --episodes 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rl.self_play_env import OpponentPolicy


def compute_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio from per-step returns.

    Args:
        returns: Array of per-step PnL deltas.
        risk_free_rate: Risk-free rate (default 0).

    Returns:
        Sharpe ratio (0.0 if std is zero).
    """
    if len(returns) == 0:
        return 0.0
    excess = returns - risk_free_rate
    std = np.std(excess)
    if std == 0:
        return 0.0
    # Annualize assuming 252 trading days, 1000 steps/day
    return float(np.mean(excess) / std * np.sqrt(252_000))


def compute_max_drawdown(cumulative_pnl: np.ndarray) -> float:
    """Compute maximum drawdown from cumulative PnL series.

    Args:
        cumulative_pnl: Array of cumulative PnL values.

    Returns:
        Maximum drawdown (non-negative value).
    """
    if len(cumulative_pnl) == 0:
        return 0.0
    peak = np.maximum.accumulate(cumulative_pnl)
    drawdown = peak - cumulative_pnl
    return float(np.max(drawdown))


def compute_inventory_utilization(inventory_series: np.ndarray) -> float:
    """Compute inventory utilization (fraction of time with active position).

    Args:
        inventory_series: Array of inventory values per step.

    Returns:
        Fraction in [0.0, 1.0] of steps with non-zero inventory.
    """
    if len(inventory_series) == 0:
        return 0.0
    return float(np.mean(inventory_series != 0))


def run_evaluation(
    model_path: str,
    num_episodes: int = 100,
    episode_length: int = 1000,
    seed: int = 42,
    opponents: list[str] | None = None,
    verbose: int = 1,
) -> dict:
    """Run evaluation of a trained model against various opponents.

    Args:
        model_path: Path to the saved model (.zip file).
        num_episodes: Number of episodes to run per opponent.
        episode_length: Steps per episode.
        seed: Random seed.
        opponents: List of opponent types to evaluate against.
                   Options: "random", "market_maker", "self" (uses model as opp).
                   Default: ["random", "market_maker"].
        verbose: Verbosity level.

    Returns:
        Dict with evaluation results per opponent type.
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        print(
            "ERROR: stable-baselines3 is required for evaluation.\n"
            "Install with: pip install 'stable-baselines3>=2.0'",
            file=sys.stderr,
        )
        sys.exit(1)

    from rl.self_play_env import SelfPlayEnv

    if opponents is None:
        opponents = ["random", "market_maker"]

    # Load model
    if not Path(model_path).exists() and not Path(model_path + ".zip").exists():
        # Try with .zip extension
        model_path_zip = model_path if model_path.endswith(".zip") else model_path + ".zip"
        if not Path(model_path_zip).exists():
            print(f"ERROR: Model not found at {model_path}", file=sys.stderr)
            sys.exit(1)

    model = PPO.load(model_path)

    results = {}

    for opp_type in opponents:
        if verbose:
            print(f"\nEvaluating against: {opp_type}")

        opponent = _create_opponent(opp_type, model_path)

        # Create self-play env with this opponent
        env = SelfPlayEnv(
            opponent_policies=[opponent],
            num_noise_traders=3,
            episode_length=episode_length,
            inventory_penalty=0.001,
            competitive_reward_weight=0.0,  # Use absolute PnL for eval
            seed=seed,
        )

        episode_results = _run_evaluation_episodes(
            model, env, opponent, num_episodes, seed
        )
        results[opp_type] = episode_results

        if verbose:
            _print_results(opp_type, episode_results)

    return results


def _create_opponent(opp_type: str, model_path: str) -> "OpponentPolicy":
    """Create an opponent policy by type.

    Args:
        opp_type: One of "random", "market_maker", "self".
        model_path: Path to the model (used for "self" opponent).

    Returns:
        OpponentPolicy instance.
    """
    from rl.self_play_env import OpponentPolicy
    import random

    if opp_type == "random":
        return OpponentPolicy(
            action_fn=lambda obs: random.randint(0, 4),
            name="random",
        )
    elif opp_type == "market_maker":
        # Market maker heuristic: buy when imbalance is positive, sell when negative
        def mm_policy(obs):
            imbalance = obs[10]  # Last element is imbalance
            inventory = obs[8] * 100  # Denormalize
            if inventory > 5:
                return 4  # sell market to reduce
            elif inventory < -5:
                return 2  # buy market to reduce
            elif imbalance > 0.2:
                return 1  # buy limit (follow the crowd)
            elif imbalance < -0.2:
                return 3  # sell limit
            else:
                return 0  # hold

        return OpponentPolicy(action_fn=mm_policy, name="market_maker")
    elif opp_type == "self":
        return OpponentPolicy(policy_path=model_path, name="self_play")
    else:
        # Default to random
        return OpponentPolicy(
            action_fn=lambda obs: random.randint(0, 4),
            name=opp_type,
        )


def _run_evaluation_episodes(
    model, env, opponent, num_episodes: int, seed: int
) -> dict:
    """Run evaluation episodes and collect metrics.

    Args:
        model: Trained PPO model.
        env: SelfPlayEnv.
        opponent: Opponent policy.
        num_episodes: Number of episodes.
        seed: Base seed.

    Returns:
        Dict with aggregated metrics.
    """
    all_pnls = []
    all_sharpes = []
    all_drawdowns = []
    all_inv_utils = []
    wins = 0

    for ep in range(num_episodes):
        env.set_opponent(opponent)
        obs, _ = env.reset(seed=seed + ep)

        step_returns = []
        cumulative_pnl = []
        inventory_series = []
        prev_pnl = 0.0

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

            current_pnl = info["pnl"]
            step_returns.append(current_pnl - prev_pnl)
            cumulative_pnl.append(current_pnl)
            inventory_series.append(info["inventory"])
            prev_pnl = current_pnl

        result = env.get_episode_result()
        if result["agent_won"]:
            wins += 1

        step_returns = np.array(step_returns)
        cumulative_pnl = np.array(cumulative_pnl)
        inventory_series = np.array(inventory_series)

        all_pnls.append(cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0.0)
        all_sharpes.append(compute_sharpe_ratio(step_returns))
        all_drawdowns.append(compute_max_drawdown(cumulative_pnl))
        all_inv_utils.append(compute_inventory_utilization(inventory_series))

    return {
        "total_pnl_mean": float(np.mean(all_pnls)),
        "total_pnl_std": float(np.std(all_pnls)),
        "sharpe_ratio_mean": float(np.mean(all_sharpes)),
        "sharpe_ratio_std": float(np.std(all_sharpes)),
        "max_drawdown_mean": float(np.mean(all_drawdowns)),
        "max_drawdown_std": float(np.std(all_drawdowns)),
        "inventory_utilization_mean": float(np.mean(all_inv_utils)),
        "inventory_utilization_std": float(np.std(all_inv_utils)),
        "win_rate": wins / max(num_episodes, 1),
        "num_episodes": num_episodes,
    }


def _print_results(opp_type: str, results: dict) -> None:
    """Print evaluation results in a formatted table.

    Args:
        opp_type: Opponent type name.
        results: Dict with evaluation metrics.
    """
    print(f"  {'Metric':<28} {'Mean':>12} {'Std':>12}")
    print(f"  {'-'*52}")
    print(
        f"  {'Total PnL':<28} {results['total_pnl_mean']:>12.2f} "
        f"{results['total_pnl_std']:>12.2f}"
    )
    print(
        f"  {'Sharpe Ratio':<28} {results['sharpe_ratio_mean']:>12.4f} "
        f"{results['sharpe_ratio_std']:>12.4f}"
    )
    print(
        f"  {'Max Drawdown':<28} {results['max_drawdown_mean']:>12.2f} "
        f"{results['max_drawdown_std']:>12.2f}"
    )
    print(
        f"  {'Inventory Utilization':<28} {results['inventory_utilization_mean']:>12.4f} "
        f"{results['inventory_utilization_std']:>12.4f}"
    )
    print(f"  {'Win Rate':<28} {results['win_rate']:>12.2%}")
    print(f"  {'Episodes':<28} {results['num_episodes']:>12d}")


def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained RL trading agent"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model (e.g., models/ppo_trader.zip)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes per opponent (default: 100)",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=1000,
        help="Steps per episode (default: 1000)",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        nargs="+",
        default=["random", "market_maker"],
        help="Opponent types to evaluate against (default: random market_maker)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity (default: 1)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  RL Trading Agent Evaluation")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Opponents: {args.opponents}")
    print("=" * 60)

    results = run_evaluation(
        model_path=args.model,
        num_episodes=args.episodes,
        episode_length=args.episode_length,
        seed=args.seed,
        opponents=args.opponents,
        verbose=args.verbose,
    )

    # Final summary table
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Opponent':<15} {'PnL':>10} {'Sharpe':>10} {'Drawdown':>10} {'Win%':>8}")
    print(f"  {'-'*53}")
    for opp, r in results.items():
        print(
            f"  {opp:<15} {r['total_pnl_mean']:>10.1f} "
            f"{r['sharpe_ratio_mean']:>10.3f} "
            f"{r['max_drawdown_mean']:>10.1f} "
            f"{r['win_rate']:>7.1%}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
