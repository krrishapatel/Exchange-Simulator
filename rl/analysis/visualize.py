"""Analysis visualization for self-play convergence.

Generates matplotlib plots or falls back to JSON output if matplotlib
is not installed. All plots are saved to the analysis_output/ directory.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rl.analysis.convergence import ConvergenceTracker

# Output directory for all plots
OUTPUT_DIR = Path("analysis_output")


def _ensure_output_dir() -> Path:
    """Create output directory if it doesn't exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _has_matplotlib() -> bool:
    """Check if matplotlib is available."""
    # find_spec rather than a try/import, so this is an availability check and
    # not an import whose unused name a linter has to be told to ignore.
    return importlib.util.find_spec("matplotlib") is not None


def plot_convergence(tracker: "ConvergenceTracker", output_dir: str | None = None) -> str:
    """Plot convergence metrics over training generations.

    Generates a line chart with:
      - Exploitability over generations
      - Policy stability (KL divergence) over generations
      - Nash gap over generations

    Args:
        tracker: ConvergenceTracker instance with recorded data.
        output_dir: Optional override for output directory.

    Returns:
        Path to the saved plot file (PNG or JSON).
    """
    out_dir = Path(output_dir) if output_dir else _ensure_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    generations = tracker.generations
    exploitability = tracker.exploitability_history
    kl_divergence = tracker.kl_divergence_history
    nash_gap = tracker.nash_gap_history

    if not _has_matplotlib():
        # Fallback: save as JSON
        data = {
            "generations": generations,
            "exploitability": exploitability,
            "kl_divergence": kl_divergence,
            "nash_gap": nash_gap,
        }
        path = str(out_dir / "convergence.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Exploitability
    axes[0].plot(generations, exploitability, "b-o", markersize=3, label="Exploitability")
    axes[0].axhline(y=tracker.config.exploitability_threshold, color="r", linestyle="--", label="Threshold")
    axes[0].set_ylabel("Exploitability")
    axes[0].set_title("Self-Play Convergence Metrics")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # KL divergence (policy stability)
    axes[1].plot(generations, kl_divergence, "g-o", markersize=3, label="KL Divergence")
    axes[1].set_ylabel("KL Divergence")
    axes[1].set_yscale("log" if any(k > 0 for k in kl_divergence) else "linear")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Nash gap
    axes[2].plot(generations, nash_gap, "r-o", markersize=3, label="Nash Gap")
    axes[2].set_xlabel("Generation")
    axes[2].set_ylabel("Nash Gap")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = str(out_dir / "convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_elo_progression(
    elo_history: list[dict[str, float]],
    player_names: list[str] | None = None,
    output_dir: str | None = None,
) -> str:
    """Plot Elo rating progression over time for each agent generation.

    Args:
        elo_history: List of dicts mapping player_name -> rating at each time step.
                     Each entry represents a snapshot of ratings after a round of matches.
        player_names: Optional subset of players to plot. If None, plots all.
        output_dir: Optional override for output directory.

    Returns:
        Path to the saved plot file (PNG or JSON).
    """
    out_dir = Path(output_dir) if output_dir else _ensure_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not elo_history:
        # Nothing to plot
        path = str(out_dir / "elo_progression.json")
        with open(path, "w") as f:
            json.dump({"error": "no data"}, f)
        return path

    # Collect all player names
    all_players = set()
    for snapshot in elo_history:
        all_players.update(snapshot.keys())

    if player_names is not None:
        all_players = all_players.intersection(set(player_names))

    all_players = sorted(all_players)
    time_steps = list(range(len(elo_history)))

    if not _has_matplotlib():
        data = {
            "time_steps": time_steps,
            "players": {},
        }
        for player in all_players:
            data["players"][player] = [
                snapshot.get(player, None) for snapshot in elo_history
            ]
        path = str(out_dir / "elo_progression.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    for player in all_players:
        ratings = [snapshot.get(player, np.nan) for snapshot in elo_history]
        ax.plot(time_steps, ratings, "-o", markersize=2, label=player)

    ax.set_xlabel("Match Round")
    ax.set_ylabel("Elo Rating")
    ax.set_title("Elo Rating Progression")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = str(out_dir / "elo_progression.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_strategy_evolution(
    checkpoints: list[dict],
    output_dir: str | None = None,
) -> str:
    """Plot heatmap of action distribution per generation.

    Shows how the agent's strategy (action probabilities) evolves
    across training generations.

    Args:
        checkpoints: List of dicts with keys:
            - 'generation': int
            - 'action_distribution': array-like of shape (num_actions,)
        output_dir: Optional override for output directory.

    Returns:
        Path to the saved plot file (PNG or JSON).
    """
    out_dir = Path(output_dir) if output_dir else _ensure_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoints:
        path = str(out_dir / "strategy_evolution.json")
        with open(path, "w") as f:
            json.dump({"error": "no data"}, f)
        return path

    generations = [cp["generation"] for cp in checkpoints]
    distributions = np.array([cp["action_distribution"] for cp in checkpoints])

    action_names = ["Hold", "Buy Limit", "Buy Market", "Sell Limit", "Sell Market"]
    # Trim action names to match actual number of actions
    num_actions = distributions.shape[1] if len(distributions.shape) > 1 else 0
    action_names = action_names[:num_actions]

    if not _has_matplotlib():
        data = {
            "generations": generations,
            "action_names": action_names,
            "distributions": distributions.tolist(),
        }
        path = str(out_dir / "strategy_evolution.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))

    im = ax.imshow(
        distributions.T,
        aspect="auto",
        cmap="YlOrRd",
        interpolation="nearest",
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Action")
    ax.set_title("Strategy Evolution (Action Distribution per Generation)")
    ax.set_xticks(range(len(generations)))
    ax.set_xticklabels(generations, rotation=45, fontsize=8)
    ax.set_yticks(range(len(action_names)))
    ax.set_yticklabels(action_names)

    plt.colorbar(im, ax=ax, label="Probability")
    plt.tight_layout()
    path = str(out_dir / "strategy_evolution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pnl_distribution(
    eval_results: list[float] | np.ndarray,
    title: str = "PnL Distribution",
    output_dir: str | None = None,
) -> str:
    """Plot histogram of PnL outcomes across episodes.

    Args:
        eval_results: List or array of PnL values from evaluation episodes.
        title: Plot title.
        output_dir: Optional override for output directory.

    Returns:
        Path to the saved plot file (PNG or JSON).
    """
    out_dir = Path(output_dir) if output_dir else _ensure_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    pnls = np.asarray(eval_results, dtype=np.float64)

    if not _has_matplotlib():
        data = {
            "pnl_values": pnls.tolist(),
            "mean": float(pnls.mean()) if len(pnls) > 0 else 0.0,
            "std": float(pnls.std()) if len(pnls) > 0 else 0.0,
            "median": float(np.median(pnls)) if len(pnls) > 0 else 0.0,
            "min": float(pnls.min()) if len(pnls) > 0 else 0.0,
            "max": float(pnls.max()) if len(pnls) > 0 else 0.0,
        }
        path = str(out_dir / "pnl_distribution.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    if len(pnls) > 0:
        n_bins = min(50, max(10, len(pnls) // 5))
        ax.hist(pnls, bins=n_bins, alpha=0.7, color="steelblue", edgecolor="white")

        # Add statistics lines
        mean_val = float(pnls.mean())
        median_val = float(np.median(pnls))
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, color="green", linestyle="-.", linewidth=2, label=f"Median: {median_val:.2f}")

    ax.set_xlabel("PnL")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = str(out_dir / "pnl_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
