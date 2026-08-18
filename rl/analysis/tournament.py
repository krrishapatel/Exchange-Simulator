"""Round-robin tournament runner for agent evaluation.

Takes N agent checkpoints + baseline agents, runs all-pairs matches,
computes Elo ratings, win rates, and average PnL per matchup.

CLI usage:
    python -m rl.analysis.tournament --checkpoints-dir models/ --episodes 50
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from rl.analysis.elo import EloRating


class AgentEntry:
    """Represents an agent participating in the tournament.

    Attributes:
        name: Human-readable agent identifier.
        action_fn: Callable that maps observation -> action integer.
        checkpoint_path: Optional path to a model checkpoint.
    """

    def __init__(
        self,
        name: str,
        action_fn: Callable[[np.ndarray], int] | None = None,
        checkpoint_path: str | Path | None = None,
    ):
        """Initialize an agent entry.

        Args:
            name: Agent identifier.
            action_fn: Function mapping observation -> action.
                       If None, loads from checkpoint_path.
            checkpoint_path: Path to SB3 model checkpoint.
        """
        self.name = name
        self.checkpoint_path = checkpoint_path
        self._action_fn = action_fn
        self._model = None

        if action_fn is None and checkpoint_path is not None:
            self._load_model(checkpoint_path)

    def _load_model(self, path: str | Path) -> None:
        """Load a stable-baselines3 model."""
        try:
            from stable_baselines3 import PPO

            self._model = PPO.load(str(path))
        except ImportError:
            raise ImportError(
                "stable-baselines3 is required to load checkpoints. "
                "Install with: pip install 'stable-baselines3>=2.0'"
            )

    def predict(self, obs: np.ndarray) -> int:
        """Predict an action given an observation.

        Args:
            obs: Observation vector.

        Returns:
            Integer action.
        """
        if self._action_fn is not None:
            return self._action_fn(obs)
        if self._model is not None:
            action, _ = self._model.predict(obs, deterministic=True)
            return int(action)
        # Fallback: random
        return int(np.random.randint(0, 5))


class MatchResult:
    """Result of a single match between two agents.

    Attributes:
        player_a: Name of player A.
        player_b: Name of player B.
        pnl_a: Total PnL of player A.
        pnl_b: Total PnL of player B.
        winner: Name of the winner (or "draw").
    """

    def __init__(self, player_a: str, player_b: str, pnl_a: float, pnl_b: float):
        self.player_a = player_a
        self.player_b = player_b
        self.pnl_a = pnl_a
        self.pnl_b = pnl_b

        if pnl_a > pnl_b:
            self.winner = player_a
        elif pnl_b > pnl_a:
            self.winner = player_b
        else:
            self.winner = "draw"

    @property
    def result_for_a(self) -> float:
        """Elo result score for player A (1.0 win, 0.5 draw, 0.0 loss)."""
        if self.winner == self.player_a:
            return 1.0
        elif self.winner == self.player_b:
            return 0.0
        return 0.5


class Tournament:
    """Round-robin tournament runner for comparing agents.

    Runs all-pairs matches between N agents over M episodes each,
    computes Elo ratings, win rates, and average PnL per matchup.

    Usage:
        tournament = Tournament(agents, episodes_per_match=50)
        results = tournament.run()
        print(results['summary_table'])
        print(results['elo_ratings'])
    """

    def __init__(
        self,
        agents: list[AgentEntry],
        episodes_per_match: int = 50,
        episode_length: int = 1000,
        seed: int = 42,
    ):
        """Initialize the tournament.

        Args:
            agents: List of AgentEntry participants.
            episodes_per_match: Number of episodes per matchup.
            episode_length: Steps per episode.
            seed: Random seed.
        """
        self.agents = agents
        self.episodes_per_match = episodes_per_match
        self.episode_length = episode_length
        self.seed = seed

        self._elo = EloRating()
        self._match_results: list[MatchResult] = []
        self._pnl_matrix: dict[tuple[str, str], list[float]] = {}
        self._win_matrix: dict[tuple[str, str], int] = {}

    @property
    def num_matchups(self) -> int:
        """Total number of unique matchups (N choose 2)."""
        n = len(self.agents)
        return n * (n - 1) // 2

    def run(self, sim_fn: Callable | None = None) -> dict:
        """Run the full round-robin tournament.

        Args:
            sim_fn: Optional simulation function with signature:
                    sim_fn(agent_a: AgentEntry, agent_b: AgentEntry,
                           num_episodes: int, episode_length: int, seed: int)
                    -> list[MatchResult]
                    If None, uses a simple PnL comparison heuristic.

        Returns:
            Dict with keys:
                - 'elo_ratings': dict of player -> rating
                - 'elo_rank': list of (player, rating) sorted desc
                - 'match_results': list of MatchResult
                - 'win_rates': dict of (player_a, player_b) -> win rate for A
                - 'avg_pnl': dict of (player_a, player_b) -> avg PnL for A
                - 'summary_table': formatted string table
                - 'num_matchups': total matchups played
        """
        if sim_fn is None:
            sim_fn = self._default_simulate

        matchups = list(itertools.combinations(range(len(self.agents)), 2))

        for i, (idx_a, idx_b) in enumerate(matchups):
            agent_a = self.agents[idx_a]
            agent_b = self.agents[idx_b]

            match_seed = self.seed + i * 1000
            results = sim_fn(
                agent_a, agent_b,
                self.episodes_per_match,
                self.episode_length,
                match_seed,
            )

            for result in results:
                self._match_results.append(result)
                self._elo.update(result.player_a, result.player_b, result.result_for_a)

                # Track PnL
                key_ab = (result.player_a, result.player_b)
                if key_ab not in self._pnl_matrix:
                    self._pnl_matrix[key_ab] = []
                self._pnl_matrix[key_ab].append(result.pnl_a)

                key_ba = (result.player_b, result.player_a)
                if key_ba not in self._pnl_matrix:
                    self._pnl_matrix[key_ba] = []
                self._pnl_matrix[key_ba].append(result.pnl_b)

                # Track wins
                if result.winner == result.player_a:
                    self._win_matrix[key_ab] = self._win_matrix.get(key_ab, 0) + 1
                elif result.winner == result.player_b:
                    self._win_matrix[key_ba] = self._win_matrix.get(key_ba, 0) + 1

        # Compute summary
        win_rates = {}
        avg_pnls = {}
        for key, pnls in self._pnl_matrix.items():
            avg_pnls[key] = float(np.mean(pnls))
            wins = self._win_matrix.get(key, 0)
            total = len(pnls)
            win_rates[key] = wins / total if total > 0 else 0.0

        summary_table = self._format_summary_table(win_rates, avg_pnls)

        return {
            "elo_ratings": self._elo.ratings(),
            "elo_rank": self._elo.rank(),
            "match_results": self._match_results,
            "win_rates": win_rates,
            "avg_pnl": avg_pnls,
            "summary_table": summary_table,
            "num_matchups": self.num_matchups,
        }

    def _default_simulate(
        self,
        agent_a: AgentEntry,
        agent_b: AgentEntry,
        num_episodes: int,
        episode_length: int,
        seed: int,
    ) -> list[MatchResult]:
        """Default simulation using random PnL for testing purposes.

        In production, this should be replaced with actual environment simulation.
        The PnL here is drawn from a fixed distribution and does not depend on
        either agent, so it exercises the bookkeeping and says nothing about who
        is stronger.

        Args:
            agent_a: First agent.
            agent_b: Second agent.
            num_episodes: Number of episodes.
            episode_length: Steps per episode.
            seed: Random seed.

        Returns:
            List of MatchResult for each episode.
        """
        rng = np.random.default_rng(seed)
        results = []

        for ep in range(num_episodes):
            pnl_a = float(rng.normal(0, 100))
            pnl_b = float(rng.normal(0, 100))

            results.append(
                MatchResult(
                    player_a=agent_a.name,
                    player_b=agent_b.name,
                    pnl_a=pnl_a,
                    pnl_b=pnl_b,
                )
            )

        return results

    def _format_summary_table(
        self,
        win_rates: dict[tuple[str, str], float],
        avg_pnls: dict[tuple[str, str], float],
    ) -> str:
        """Format a summary table of tournament results.

        Args:
            win_rates: Dict of (player_a, player_b) -> win rate for A.
            avg_pnls: Dict of (player_a, player_b) -> avg PnL for A.

        Returns:
            Formatted string table.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("  TOURNAMENT RESULTS")
        lines.append("=" * 70)

        # Elo rankings
        lines.append("\n  Elo Rankings:")
        lines.append(f"  {'Rank':<6}{'Agent':<20}{'Rating':<12}{'Games':<8}")
        lines.append(f"  {'-'*46}")
        for rank, (name, rating) in enumerate(self._elo.rank(), 1):
            games = self._elo.get_games_played(name)
            lines.append(f"  {rank:<6}{name:<20}{rating:<12.1f}{games:<8}")

        # Head-to-head
        lines.append("\n  Head-to-Head Win Rates:")
        lines.append(f"  {'Matchup':<30}{'Win Rate':<12}{'Avg PnL':<12}")
        lines.append(f"  {'-'*54}")
        for (a, b), rate in sorted(win_rates.items()):
            pnl = avg_pnls.get((a, b), 0.0)
            lines.append(f"  {a} vs {b:<15}{rate:<12.2%}{pnl:<12.1f}")

        lines.append("=" * 70)
        return "\n".join(lines)

    @property
    def elo(self) -> EloRating:
        """Access the underlying EloRating instance."""
        return self._elo


def main():
    """CLI entry point for running a tournament."""
    parser = argparse.ArgumentParser(
        description="Run round-robin tournament between agent checkpoints"
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="models/",
        help="Directory containing model checkpoints (default: models/)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Episodes per matchup (default: 50)",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=1000,
        help="Steps per episode (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        default=True,
        help="Include random and market-maker baselines (default: True)",
    )
    args = parser.parse_args()

    # Discover checkpoints
    checkpoint_dir = Path(args.checkpoints_dir)
    agents: list[AgentEntry] = []

    if checkpoint_dir.exists():
        # Look for .zip files (SB3 format)
        for cp_path in sorted(checkpoint_dir.glob("*.zip")):
            name = cp_path.stem
            try:
                agent = AgentEntry(name=name, checkpoint_path=cp_path)
                agents.append(agent)
                print(f"  Loaded: {name}")
            except ImportError:
                print(f"  Skipped (no SB3): {name}", file=sys.stderr)

    # Add baselines
    if args.include_baselines:
        import random

        agents.append(
            AgentEntry(
                name="random_baseline",
                action_fn=lambda obs: random.randint(0, 4),
            )
        )

        def mm_policy(obs):
            imbalance = obs[10] if len(obs) > 10 else 0.0
            inventory = obs[8] * 100 if len(obs) > 8 else 0.0
            if inventory > 5:
                return 4
            elif inventory < -5:
                return 2
            elif imbalance > 0.2:
                return 1
            elif imbalance < -0.2:
                return 3
            return 0

        agents.append(AgentEntry(name="market_maker", action_fn=mm_policy))

    if len(agents) < 2:
        print("ERROR: Need at least 2 agents for a tournament.", file=sys.stderr)
        sys.exit(1)

    print(f"\nRunning tournament with {len(agents)} agents:")
    print(f"  Episodes per match: {args.episodes}")
    print(f"  Episode length: {args.episode_length}")
    print(f"  Total matchups: {len(agents) * (len(agents) - 1) // 2}")
    print()

    tournament = Tournament(
        agents=agents,
        episodes_per_match=args.episodes,
        episode_length=args.episode_length,
        seed=args.seed,
    )
    results = tournament.run()

    print(results["summary_table"])


if __name__ == "__main__":
    main()
