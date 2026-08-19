"""Self-play convergence analysis and visualization tools.

Provides:
  - ConvergenceTracker: monitors training progress toward Nash equilibrium
  - EloRating: Elo rating system for agent pool evaluation
  - Tournament: round-robin tournament runner
  - Visualization functions for convergence, Elo, strategy, and PnL
"""

from rl.analysis.convergence import ConvergenceTracker
from rl.analysis.elo import EloRating
from rl.analysis.league import run_league
from rl.analysis.tournament import Tournament
from rl.analysis.visualize import (
    plot_convergence,
    plot_elo_progression,
    plot_pnl_distribution,
    plot_strategy_evolution,
)

__all__ = [
    "ConvergenceTracker",
    "EloRating",
    "Tournament",
    "run_league",
    "plot_convergence",
    "plot_elo_progression",
    "plot_pnl_distribution",
    "plot_strategy_evolution",
]
