"""Self-play training orchestrator with league-style opponent pool.

Maintains a pool of opponent checkpoints and trains PPO against
past versions of itself. Implements league training where:
  - Every N episodes, current policy is snapshot into the pool
  - Opponents are sampled: 50% most recent, 50% uniform
  - Win rates against each generation are logged

Usage:
    python rl/self_play.py --timesteps 500000 --pool-size 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path



class OpponentPool:
    """Manages a pool of opponent policy checkpoints.

    Stores up to max_size frozen checkpoints. Provides sampling
    with bias toward recent opponents.
    """

    def __init__(self, max_size: int = 10, save_dir: str = "models/opponent_pool"):
        """Initialize the opponent pool.

        Args:
            max_size: Maximum number of checkpoints to retain.
            save_dir: Directory to save checkpoint files.
        """
        self.max_size = max_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._checkpoints: list[Path] = []
        self._generation: int = 0
        self._win_rates: dict[int, list[bool]] = defaultdict(list)

    @property
    def size(self) -> int:
        """Number of checkpoints in the pool."""
        return len(self._checkpoints)

    @property
    def generation(self) -> int:
        """Current generation counter."""
        return self._generation

    def add_checkpoint(self, model) -> Path:
        """Save a model checkpoint to the pool.

        If pool is full, removes the oldest checkpoint (FIFO).

        Args:
            model: A stable-baselines3 PPO model to save.

        Returns:
            Path to the saved checkpoint.
        """
        self._generation += 1
        path = self.save_dir / f"gen_{self._generation:04d}"
        model.save(str(path))

        self._checkpoints.append(path)

        # Evict oldest if over capacity
        if len(self._checkpoints) > self.max_size:
            old_path = self._checkpoints.pop(0)
            # Clean up old file
            old_file = Path(str(old_path) + ".zip")
            if old_file.exists():
                old_file.unlink()

        return path

    def sample(self, rng=None) -> tuple[Path, int]:
        """Sample a checkpoint from the pool.

        50% chance of most recent, 50% uniform over all.

        Args:
            rng: Optional random.Random instance.

        Returns:
            Tuple of (checkpoint_path, generation_index).
        """
        import random

        if rng is None:
            rng = random.Random()

        if len(self._checkpoints) == 0:
            raise ValueError("No checkpoints in pool")

        if len(self._checkpoints) == 1:
            return self._checkpoints[0], 0

        if rng.random() < 0.5:
            # Most recent
            idx = len(self._checkpoints) - 1
        else:
            # Uniform
            idx = rng.randint(0, len(self._checkpoints) - 1)

        return self._checkpoints[idx], idx

    def record_result(self, generation_idx: int, agent_won: bool) -> None:
        """Record a win/loss result against a specific generation.

        Args:
            generation_idx: Index of the opponent in the pool.
            agent_won: Whether the training agent won.
        """
        self._win_rates[generation_idx].append(agent_won)

    def get_win_rates(self) -> dict[int, float]:
        """Get win rates against each generation.

        Returns:
            Dict mapping generation index to win rate (0.0 - 1.0).
        """
        rates = {}
        for gen_idx, results in self._win_rates.items():
            if results:
                rates[gen_idx] = sum(results) / len(results)
        return rates

    def get_checkpoint_paths(self) -> list[Path]:
        """Return all checkpoint paths in the pool."""
        return list(self._checkpoints)


def make_self_play_env(opponent_pool: OpponentPool | None = None, seed: int = 42):
    """Create a SelfPlayEnv with opponents from the pool.

    Args:
        opponent_pool: Pool of opponent checkpoints.
        seed: Random seed.

    Returns:
        A SelfPlayEnv instance.
    """
    from rl.self_play_env import OpponentPolicy, SelfPlayEnv

    opponents = []

    # Always include a random baseline
    import random

    opponents.append(
        OpponentPolicy(
            action_fn=lambda obs: random.randint(0, 4),
            name="random_baseline",
        )
    )

    # Load checkpoints from pool if available
    if opponent_pool is not None and opponent_pool.size > 0:
        for i, path in enumerate(opponent_pool.get_checkpoint_paths()):
            try:
                opp = OpponentPolicy(
                    policy_path=path,
                    name=f"gen_{i}",
                )
                opponents.append(opp)
            except Exception:
                pass  # Skip failed loads

    env = SelfPlayEnv(
        opponent_policies=opponents,
        num_noise_traders=3,
        episode_length=1000,
        inventory_penalty=0.001,
        competitive_reward_weight=0.5,
        seed=seed,
    )
    return env


class SelfPlayCallback:
    """Callback for self-play training that snapshots and updates opponents.

    Integrates with stable-baselines3's callback system.
    """

    def __init__(
        self,
        opponent_pool: OpponentPool,
        snapshot_interval: int = 10000,
        env=None,
        verbose: int = 1,
    ):
        """Initialize the callback.

        Args:
            opponent_pool: Pool to save checkpoints to.
            snapshot_interval: Steps between snapshots.
            env: The SelfPlayEnv to update opponents on.
            verbose: Verbosity level.
        """
        self.opponent_pool = opponent_pool
        self.snapshot_interval = snapshot_interval
        self.env = env
        self.verbose = verbose
        self._n_calls = 0
        self._last_snapshot = 0

    def on_step(self, model) -> None:
        """Called each training step.

        Args:
            model: The current PPO model being trained.
        """
        self._n_calls += 1

        if self._n_calls - self._last_snapshot >= self.snapshot_interval:
            self._last_snapshot = self._n_calls
            path = self.opponent_pool.add_checkpoint(model)
            if self.verbose:
                print(
                    f"  [SelfPlay] Snapshot saved: gen {self.opponent_pool.generation} "
                    f"(pool size: {self.opponent_pool.size}) -> {path}"
                )

            # Update the env's opponent pool
            if self.env is not None:
                self._refresh_env_opponents()

    def _refresh_env_opponents(self) -> None:
        """Refresh the environment's opponent pool from checkpoints."""
        from rl.self_play_env import OpponentPolicy
        import random

        opponents = [
            OpponentPolicy(
                action_fn=lambda obs: random.randint(0, 4),
                name="random_baseline",
            )
        ]

        for i, path in enumerate(self.opponent_pool.get_checkpoint_paths()):
            try:
                opp = OpponentPolicy(policy_path=path, name=f"gen_{i}")
                opponents.append(opp)
            except Exception:
                pass

        self.env.set_opponent_pool(opponents)


def train_self_play(
    total_timesteps: int = 500_000,
    pool_size: int = 10,
    snapshot_interval: int = 10_000,
    model_save_path: str = "models/ppo_self_play",
    pool_dir: str = "models/opponent_pool",
    seed: int = 42,
    verbose: int = 1,
) -> dict:
    """Run self-play training.

    Args:
        total_timesteps: Total training timesteps.
        pool_size: Max opponent pool size.
        snapshot_interval: Steps between policy snapshots.
        model_save_path: Where to save the final model.
        pool_dir: Directory for opponent checkpoint pool.
        seed: Random seed.
        verbose: Verbosity (0=silent, 1=progress).

    Returns:
        Dict with training stats (win_rates, generations, etc).
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError:
        print(
            "ERROR: stable-baselines3 is required for self-play training.\n"
            "Install with: pip install 'stable-baselines3>=2.0'",
            file=sys.stderr,
        )
        sys.exit(1)

    # Create output dirs
    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)

    # Initialize opponent pool
    pool = OpponentPool(max_size=pool_size, save_dir=pool_dir)

    # Create environment
    env = make_self_play_env(opponent_pool=pool, seed=seed)

    # Create SB3-compatible callback
    sp_callback = SelfPlayCallback(
        opponent_pool=pool,
        snapshot_interval=snapshot_interval,
        env=env,
        verbose=verbose,
    )

    class SB3SelfPlayCallback(BaseCallback):
        """Wraps SelfPlayCallback for stable-baselines3."""

        def __init__(self, sp_cb, verbose=0):
            super().__init__(verbose)
            self.sp_cb = sp_cb

        def _on_step(self) -> bool:
            self.sp_cb.on_step(self.model)
            return True

    sb3_callback = SB3SelfPlayCallback(sp_callback, verbose=verbose)

    if verbose:
        print("Starting self-play training:")
        print(f"  Total timesteps: {total_timesteps}")
        print(f"  Pool size: {pool_size}")
        print(f"  Snapshot interval: {snapshot_interval}")
        print(f"  Seed: {seed}")

    # Initialize PPO
    model = PPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        seed=seed,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
    )

    # Save initial model as gen 0
    pool.add_checkpoint(model)
    sp_callback._refresh_env_opponents()

    # Train
    start_time = time.time()
    model.learn(total_timesteps=total_timesteps, callback=sb3_callback)
    elapsed = time.time() - start_time

    # Save final model
    model.save(model_save_path)

    # Run evaluation episodes to compute win rates
    if verbose:
        print(f"\nTraining complete in {elapsed:.1f}s")
        print(f"Final model saved to: {model_save_path}.zip")
        print(f"Opponent pool generations: {pool.generation}")
        print("\nRunning evaluation against pool...")

    # Evaluate against each generation
    win_rates = _evaluate_against_pool(model, pool, env, num_episodes=5)

    if verbose:
        print("\nWin rates vs opponent generations:")
        for gen_name, rate in win_rates.items():
            print(f"  vs {gen_name}: {rate:.1%}")

    return {
        "model_path": f"{model_save_path}.zip",
        "generations": pool.generation,
        "elapsed_seconds": elapsed,
        "win_rates": win_rates,
    }


def _evaluate_against_pool(
    model, pool: OpponentPool, env, num_episodes: int = 5
) -> dict[str, float]:
    """Evaluate trained model against each opponent in the pool.

    Args:
        model: Trained PPO model.
        pool: Opponent pool.
        env: SelfPlayEnv instance.
        num_episodes: Episodes per opponent.

    Returns:
        Dict mapping opponent name to win rate.
    """
    from rl.self_play_env import OpponentPolicy
    import random

    results = {}

    # vs random baseline
    random_opp = OpponentPolicy(
        action_fn=lambda obs: random.randint(0, 4),
        name="random_baseline",
    )
    wins = _run_episodes(model, env, random_opp, num_episodes)
    results["random_baseline"] = wins / max(num_episodes, 1)

    # vs each generation
    for i, path in enumerate(pool.get_checkpoint_paths()):
        try:
            opp = OpponentPolicy(policy_path=path, name=f"gen_{i}")
            wins = _run_episodes(model, env, opp, num_episodes)
            results[f"gen_{i}"] = wins / max(num_episodes, 1)
        except Exception:
            pass

    return results


def _run_episodes(model, env, opponent, num_episodes: int) -> int:
    """Run episodes and count wins.

    Args:
        model: Trained model.
        env: SelfPlayEnv.
        opponent: OpponentPolicy to evaluate against.
        num_episodes: Number of episodes.

    Returns:
        Number of wins.
    """
    wins = 0
    for ep in range(num_episodes):
        env.set_opponent(opponent)
        obs, _ = env.reset(seed=ep + 1000)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
        result = env.get_episode_result()
        if result["agent_won"]:
            wins += 1
    return wins


def main():
    """CLI entry point for self-play training."""
    parser = argparse.ArgumentParser(
        description="Self-play RL training with league-style opponent pool"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500000)",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=10,
        help="Max opponent pool size (default: 10)",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=10_000,
        help="Steps between policy snapshots (default: 10000)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/ppo_self_play",
        help="Path to save final model (default: models/ppo_self_play)",
    )
    parser.add_argument(
        "--pool-dir",
        type=str,
        default="models/opponent_pool",
        help="Directory for opponent pool (default: models/opponent_pool)",
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
        help="Verbosity level (default: 1)",
    )
    args = parser.parse_args()

    results = train_self_play(
        total_timesteps=args.timesteps,
        pool_size=args.pool_size,
        snapshot_interval=args.snapshot_interval,
        model_save_path=args.model_path,
        pool_dir=args.pool_dir,
        seed=args.seed,
        verbose=args.verbose,
    )

    print(f"\nDone. Model: {results['model_path']}")


if __name__ == "__main__":
    main()
