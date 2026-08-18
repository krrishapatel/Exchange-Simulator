"""Elo rating system for agent pool evaluation.

Implements standard Elo with decaying K-factor and optional
bootstrapped confidence intervals. Used to rank agents in the
self-play opponent pool by relative skill.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class EloConfig:
    """Configuration for the Elo rating system.

    Attributes:
        initial_rating: Starting Elo for new agents.
        base_k: Initial K-factor (sensitivity to results).
        k_decay: K-factor decay rate per game played. K = base_k / (1 + k_decay * games).
        min_k: Minimum K-factor floor.
        scale: Elo scale denominator (standard is 400).
    """

    initial_rating: float = 1500.0
    base_k: float = 32.0
    k_decay: float = 0.01
    min_k: float = 8.0
    scale: float = 400.0


class EloRating:
    """Elo rating system for ranking agents in the self-play pool.

    Maintains ratings for all agents and updates them based on
    head-to-head match results. Supports win/loss/draw outcomes
    and computes confidence intervals via bootstrapping.

    Usage:
        elo = EloRating()
        elo.update("agent_gen5", "agent_gen3", 1.0)  # gen5 wins
        elo.update("agent_gen5", "agent_gen4", 0.5)  # draw
        print(elo.ratings())
        print(elo.rank())
    """

    def __init__(self, config: EloConfig | None = None):
        """Initialize the Elo rating system.

        Args:
            config: Elo configuration parameters. Uses defaults if None.
        """
        self.config = config or EloConfig()

        self._ratings: dict[str, float] = {}
        self._games_played: dict[str, int] = defaultdict(int)
        self._history: list[dict] = []  # Full history of all updates

    def _ensure_player(self, player: str) -> None:
        """Ensure a player exists in the rating table.

        Args:
            player: Player identifier.
        """
        if player not in self._ratings:
            self._ratings[player] = self.config.initial_rating

    def _get_k_factor(self, player: str) -> float:
        """Get the current K-factor for a player (decays with games played).

        K = base_k / (1 + k_decay * games_played), floored at min_k.

        Args:
            player: Player identifier.

        Returns:
            Current K-factor for this player.
        """
        games = self._games_played[player]
        k = self.config.base_k / (1 + self.config.k_decay * games)
        return max(k, self.config.min_k)

    def expected_score(self, player_a: str, player_b: str) -> float:
        """Compute expected score for player_a against player_b.

        E_a = 1 / (1 + 10^((R_b - R_a) / 400))

        Args:
            player_a: First player identifier.
            player_b: Second player identifier.

        Returns:
            Expected score for player_a (between 0 and 1).
        """
        self._ensure_player(player_a)
        self._ensure_player(player_b)

        r_a = self._ratings[player_a]
        r_b = self._ratings[player_b]

        exponent = (r_b - r_a) / self.config.scale
        return 1.0 / (1.0 + 10.0**exponent)

    def update(self, player_a: str, player_b: str, result: float) -> tuple[float, float]:
        """Update ratings based on a match result.

        Args:
            player_a: First player identifier.
            player_b: Second player identifier.
            result: Score for player_a: 1.0 = win, 0.5 = draw, 0.0 = loss.

        Returns:
            Tuple of (new_rating_a, new_rating_b).

        Raises:
            ValueError: If result is not in [0.0, 1.0].
        """
        if not 0.0 <= result <= 1.0:
            raise ValueError(f"Result must be in [0.0, 1.0], got {result}")

        self._ensure_player(player_a)
        self._ensure_player(player_b)

        e_a = self.expected_score(player_a, player_b)
        e_b = 1.0 - e_a

        k_a = self._get_k_factor(player_a)
        k_b = self._get_k_factor(player_b)

        # Update ratings
        old_a = self._ratings[player_a]
        old_b = self._ratings[player_b]

        self._ratings[player_a] = old_a + k_a * (result - e_a)
        self._ratings[player_b] = old_b + k_b * ((1.0 - result) - e_b)

        # Track games played
        self._games_played[player_a] += 1
        self._games_played[player_b] += 1

        # Record history
        self._history.append(
            {
                "player_a": player_a,
                "player_b": player_b,
                "result": result,
                "rating_a_before": old_a,
                "rating_b_before": old_b,
                "rating_a_after": self._ratings[player_a],
                "rating_b_after": self._ratings[player_b],
            }
        )

        return self._ratings[player_a], self._ratings[player_b]

    def ratings(self) -> dict[str, float]:
        """Get current ratings for all players.

        Returns:
            Dict mapping player name to current Elo rating.
        """
        return dict(self._ratings)

    def rank(self) -> list[tuple[str, float]]:
        """Get players ranked by rating (highest first).

        Returns:
            List of (player_name, rating) tuples sorted descending by rating.
        """
        return sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)

    def get_rating(self, player: str) -> float:
        """Get a specific player's rating.

        Args:
            player: Player identifier.

        Returns:
            Current Elo rating.
        """
        self._ensure_player(player)
        return self._ratings[player]

    def get_games_played(self, player: str) -> int:
        """Get number of games played by a player.

        Args:
            player: Player identifier.

        Returns:
            Number of games played.
        """
        return self._games_played[player]

    @property
    def history(self) -> list[dict]:
        """Full history of rating updates."""
        return list(self._history)

    def rating_history_for(self, player: str) -> list[float]:
        """Get the rating history for a specific player over time.

        Args:
            player: Player identifier.

        Returns:
            List of ratings after each game involving this player.
        """
        ratings = []
        for record in self._history:
            if record["player_a"] == player:
                ratings.append(record["rating_a_after"])
            elif record["player_b"] == player:
                ratings.append(record["rating_b_after"])
        return ratings

    def confidence_intervals(
        self,
        num_bootstrap: int = 1000,
        confidence: float = 0.95,
        seed: int = 42,
    ) -> dict[str, tuple[float, float]]:
        """Compute confidence intervals for ratings via bootstrapping.

        Resamples the match history and recomputes ratings to estimate
        uncertainty in the rating estimates.

        Args:
            num_bootstrap: Number of bootstrap samples.
            confidence: Confidence level (e.g. 0.95 for 95% CI).
            seed: Random seed for reproducibility.

        Returns:
            Dict mapping player to (lower_bound, upper_bound) tuple.
        """
        if not self._history:
            return {p: (r, r) for p, r in self._ratings.items()}

        rng = np.random.default_rng(seed)
        n_matches = len(self._history)
        players = list(self._ratings.keys())

        # Collect bootstrap rating samples
        bootstrap_ratings: dict[str, list[float]] = {p: [] for p in players}

        for _ in range(num_bootstrap):
            # Resample match history with replacement
            indices = rng.integers(0, n_matches, size=n_matches)

            # Create a fresh Elo system and replay resampled matches
            temp_elo = EloRating(config=self.config)
            for idx in indices:
                record = self._history[idx]
                temp_elo.update(
                    record["player_a"],
                    record["player_b"],
                    record["result"],
                )

            # Record final ratings
            for p in players:
                if p in temp_elo._ratings:
                    bootstrap_ratings[p].append(temp_elo._ratings[p])
                else:
                    bootstrap_ratings[p].append(self.config.initial_rating)

        # Compute percentiles
        alpha = (1 - confidence) / 2
        intervals = {}
        for p in players:
            samples = np.array(bootstrap_ratings[p])
            lower = float(np.percentile(samples, alpha * 100))
            upper = float(np.percentile(samples, (1 - alpha) * 100))
            intervals[p] = (lower, upper)

        return intervals

    def total_rating_sum(self) -> float:
        """Compute total sum of all ratings.

        In standard Elo, the sum of ratings should be approximately
        conserved (N * initial_rating) when K-factors are equal.

        Returns:
            Sum of all current ratings.
        """
        return sum(self._ratings.values())
