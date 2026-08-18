"""Convergence metrics tracker for self-play training.

Monitors training progress toward Nash equilibrium by tracking:
  - Exploitability (how much a best-response can extract)
  - Policy stability (KL divergence between consecutive checkpoints)
  - Win rate plateau (moving average slope detection)
  - Nash gap estimate (upper bound on distance to Nash)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConvergenceConfig:
    """Configuration for convergence detection.

    Attributes:
        exploitability_threshold: Convergence threshold for exploitability.
        win_rate_slope_epsilon: Maximum slope to be considered a plateau.
        plateau_window: Number of checkpoints for moving average.
        consecutive_required: How many consecutive checkpoints must satisfy criteria.
        kl_threshold: Optional threshold for policy stability (KL divergence).
    """

    exploitability_threshold: float = 0.05
    win_rate_slope_epsilon: float = 0.001
    plateau_window: int = 10
    consecutive_required: int = 5
    kl_threshold: float | None = None


class ConvergenceTracker:
    """Tracks convergence of self-play training toward Nash equilibrium.

    Monitors multiple convergence signals:
      1. Exploitability: reward a best-response can extract vs current policy
      2. Policy stability: KL divergence between consecutive policy checkpoints
      3. Win rate plateau: slope of win rate moving average falls below epsilon
      4. Nash gap: max(exploitability_as_p1, exploitability_as_p2)

    Usage:
        tracker = ConvergenceTracker()
        for gen in range(num_generations):
            metrics = train_and_evaluate(...)
            tracker.update(checkpoint_path, metrics)
            if tracker.is_converged():
                break
    """

    def __init__(self, config: ConvergenceConfig | None = None):
        """Initialize the convergence tracker.

        Args:
            config: Configuration for convergence criteria. Uses defaults if None.
        """
        self.config = config or ConvergenceConfig()

        # History of metrics per checkpoint
        self._exploitability_history: list[float] = []
        self._exploitability_p1_history: list[float] = []
        self._exploitability_p2_history: list[float] = []
        self._kl_divergence_history: list[float] = []
        self._win_rate_history: list[float] = []
        self._nash_gap_history: list[float] = []
        self._checkpoint_paths: list[str] = []
        self._generations: list[int] = []
        self._generation_counter: int = 0

    @property
    def num_checkpoints(self) -> int:
        """Number of checkpoints tracked so far."""
        return len(self._checkpoint_paths)

    def update(self, checkpoint_path: str, metrics: dict) -> None:
        """Record metrics for a new checkpoint.

        Args:
            checkpoint_path: Path to the policy checkpoint file.
            metrics: Dict containing one or more of:
                - 'exploitability': float, overall exploitability
                - 'exploitability_p1': float, exploitability as player 1
                - 'exploitability_p2': float, exploitability as player 2
                - 'kl_divergence': float, KL div from previous policy
                - 'win_rate': float, win rate against opponent pool
                - 'action_distribution': np.ndarray, for computing KL if not provided
        """
        self._generation_counter += 1
        self._generations.append(self._generation_counter)
        self._checkpoint_paths.append(checkpoint_path)

        # Exploitability
        exploit = metrics.get("exploitability", 0.0)
        exploit_p1 = metrics.get("exploitability_p1", exploit)
        exploit_p2 = metrics.get("exploitability_p2", exploit)
        self._exploitability_history.append(exploit)
        self._exploitability_p1_history.append(exploit_p1)
        self._exploitability_p2_history.append(exploit_p2)

        # Nash gap: max(exploitability_as_p1, exploitability_as_p2)
        nash_gap = max(exploit_p1, exploit_p2)
        self._nash_gap_history.append(nash_gap)

        # KL divergence
        kl = metrics.get("kl_divergence", 0.0)
        if kl == 0.0 and "action_distribution" in metrics:
            kl = self._compute_kl_from_distribution(metrics["action_distribution"])
        self._kl_divergence_history.append(kl)

        # Win rate
        win_rate = metrics.get("win_rate", 0.5)
        self._win_rate_history.append(win_rate)

    def _compute_kl_from_distribution(self, current_dist: np.ndarray) -> float:
        """Compute KL divergence from action distribution vs previous.

        If no previous distribution exists, returns 0.0.

        Args:
            current_dist: Probability distribution over actions.

        Returns:
            KL divergence (non-negative).
        """
        if not hasattr(self, "_prev_action_dist") or self._prev_action_dist is None:
            self._prev_action_dist = current_dist
            return 0.0

        prev = np.asarray(self._prev_action_dist, dtype=np.float64)
        curr = np.asarray(current_dist, dtype=np.float64)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        prev = np.clip(prev, eps, 1.0)
        curr = np.clip(curr, eps, 1.0)

        # Normalize
        prev = prev / prev.sum()
        curr = curr / curr.sum()

        kl = float(np.sum(curr * np.log(curr / prev)))
        self._prev_action_dist = current_dist
        return max(kl, 0.0)

    def win_rate_slope(self, window: int | None = None) -> float:
        """Compute the slope of the win rate moving average.

        Uses linear regression over the last `window` checkpoints.

        Args:
            window: Number of recent checkpoints to consider.
                    Defaults to config.plateau_window.

        Returns:
            Slope of the linear fit (rate of change per checkpoint).
        """
        window = window or self.config.plateau_window
        if len(self._win_rate_history) < 2:
            return float("inf")

        recent = self._win_rate_history[-window:]
        if len(recent) < 2:
            return float("inf")

        x = np.arange(len(recent), dtype=np.float64)
        y = np.array(recent, dtype=np.float64)

        # Linear regression: slope = cov(x,y) / var(x)
        x_mean = x.mean()
        y_mean = y.mean()
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return 0.0
        return float(numerator / denominator)

    def is_converged(self) -> bool:
        """Check if training has converged based on configured criteria.

        Convergence requires ALL of:
          1. Exploitability < threshold for N consecutive checkpoints
          2. Win rate slope < epsilon for N consecutive checkpoints

        Optionally (if kl_threshold is set):
          3. KL divergence < kl_threshold for N consecutive checkpoints

        Returns:
            True if all convergence criteria are met.
        """
        n = self.config.consecutive_required

        if self.num_checkpoints < n:
            return False

        # Check exploitability
        recent_exploit = self._exploitability_history[-n:]
        exploit_ok = all(
            e < self.config.exploitability_threshold for e in recent_exploit
        )
        if not exploit_ok:
            return False

        # Check win rate plateau
        # Compute slope for the last N windows
        slopes_ok = True
        for i in range(n):
            end_idx = self.num_checkpoints - i
            if end_idx < 2:
                slopes_ok = False
                break
            recent = self._win_rate_history[:end_idx]
            window = min(self.config.plateau_window, len(recent))
            subset = recent[-window:]
            if len(subset) < 2:
                slopes_ok = False
                break
            x = np.arange(len(subset), dtype=np.float64)
            y = np.array(subset, dtype=np.float64)
            x_mean = x.mean()
            y_mean = y.mean()
            num = np.sum((x - x_mean) * (y - y_mean))
            den = np.sum((x - x_mean) ** 2)
            slope = num / den if den != 0 else 0.0
            if abs(slope) > self.config.win_rate_slope_epsilon:
                slopes_ok = False
                break

        if not slopes_ok:
            return False

        # Optionally check KL divergence
        if self.config.kl_threshold is not None:
            recent_kl = self._kl_divergence_history[-n:]
            kl_ok = all(k < self.config.kl_threshold for k in recent_kl)
            if not kl_ok:
                return False

        return True

    def summary(self) -> dict:
        """Return a summary of the current convergence state.

        Returns:
            Dict with current metric values and convergence status.
        """
        result = {
            "num_checkpoints": self.num_checkpoints,
            "is_converged": self.is_converged(),
            "win_rate_slope": self.win_rate_slope(),
        }

        if self._exploitability_history:
            result["latest_exploitability"] = self._exploitability_history[-1]
            result["mean_exploitability"] = float(
                np.mean(self._exploitability_history)
            )

        if self._nash_gap_history:
            result["latest_nash_gap"] = self._nash_gap_history[-1]
            result["mean_nash_gap"] = float(np.mean(self._nash_gap_history))

        if self._kl_divergence_history:
            result["latest_kl_divergence"] = self._kl_divergence_history[-1]

        if self._win_rate_history:
            result["latest_win_rate"] = self._win_rate_history[-1]
            result["mean_win_rate"] = float(np.mean(self._win_rate_history))

        return result

    @property
    def exploitability_history(self) -> list[float]:
        """Full history of exploitability values."""
        return list(self._exploitability_history)

    @property
    def kl_divergence_history(self) -> list[float]:
        """Full history of KL divergence values."""
        return list(self._kl_divergence_history)

    @property
    def win_rate_history(self) -> list[float]:
        """Full history of win rate values."""
        return list(self._win_rate_history)

    @property
    def nash_gap_history(self) -> list[float]:
        """Full history of Nash gap estimates."""
        return list(self._nash_gap_history)

    @property
    def generations(self) -> list[int]:
        """List of generation numbers."""
        return list(self._generations)
