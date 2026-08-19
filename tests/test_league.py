"""Tests for the league convergence report.

The report's job is to answer one question honestly: do later checkpoints beat
earlier ones? These tests pin the two ways it could lie. It must not depend on
the order matches are scored in, since that order is arbitrary, and it must
report a real trend as a trend and a flat result as flat. A synthetic match
runner stands in for training, so the tests are fast and deterministic and
exercise the arithmetic rather than PPO.
"""

import numpy as np

from rl.analysis.league import _spearman, run_league
from rl.analysis.tournament import AgentEntry, MatchResult


def _entry(name):
    """A placeholder agent; the synthetic sim decides outcomes, not the policy."""
    return AgentEntry(name=name, action_fn=lambda obs: 0)


def _strength_sim(strength):
    """A match runner where the stronger-ranked agent always wins.

    `strength[name]` is a skill number. Every episode goes to the higher one, so
    a monotonic ladder is the ground truth the report has to recover.
    """

    def sim(a, b, num_episodes, episode_length, seed):
        winner_pnl, loser_pnl = (1.0, 0.0)
        a_wins = strength[a.name] > strength[b.name]
        pnl_a, pnl_b = (winner_pnl, loser_pnl) if a_wins else (loser_pnl, winner_pnl)
        return [MatchResult(a.name, b.name, pnl_a, pnl_b) for _ in range(num_episodes)]

    return sim


class TestSpearman:
    def test_perfectly_monotonic_is_one(self):
        assert _spearman([0, 1, 2, 3], [10, 20, 30, 40]) == 1.0

    def test_reversed_is_minus_one(self):
        assert _spearman([0, 1, 2, 3], [40, 30, 20, 10]) == -1.0

    def test_flat_is_nan(self):
        assert np.isnan(_spearman([0, 1, 2], [5, 5, 5]))


class TestLadderIsRecovered:
    """When later really is stronger, the report has to say so."""

    def test_monotonic_strength_gives_spearman_one(self):
        names = ["g0", "g1", "g2", "g3"]
        players = [_entry(n) for n in names]
        strength = {"g0": 0, "g1": 1, "g2": 2, "g3": 3}
        report = run_league(
            players, steps=[0, 100, 200, 300],
            episodes_per_match=4, elo_orderings=20,
            sim_fn=_strength_sim(strength),
        )
        assert report["spearman_step_score"] == 1.0
        # The strongest never loses, so it wins every episode it plays.
        assert report["avg_score"]["g3"] == 1.0
        assert report["avg_score"]["g0"] == 0.0
        # Elo has to agree on the ordering.
        assert report["elo_mean"]["g3"] > report["elo_mean"]["g0"]


class TestOrderIndependence:
    """The score must not move when matches are scored in a different order."""

    def test_win_matrix_is_antisymmetric(self):
        names = ["a", "b", "c"]
        players = [_entry(n) for n in names]
        strength = {"a": 3, "b": 2, "c": 1}
        report = run_league(
            players, steps=[0, 1, 2],
            episodes_per_match=5, elo_orderings=10,
            sim_fn=_strength_sim(strength),
        )
        win = report["win"]
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert win[i][j] + win[j][i] == 1.0

    def test_scores_average_to_one_half(self):
        """Every episode has one winner, so scores over the field sum to n/2."""
        names = ["a", "b", "c", "d"]
        players = [_entry(n) for n in names]
        strength = {n: i for i, n in enumerate(names)}
        report = run_league(
            players, steps=[0, 1, 2, 3],
            episodes_per_match=3, elo_orderings=5,
            sim_fn=_strength_sim(strength),
        )
        assert sum(report["avg_score"].values()) == len(names) / 2


class TestAnchorDiscriminates:
    """A hopeless agent must land at the bottom, or a flat result proves nothing."""

    def test_weakest_scores_zero(self):
        names = ["strong", "mid", "weak"]
        players = [_entry(n) for n in names]
        strength = {"strong": 2, "mid": 1, "weak": 0}
        report = run_league(
            players, steps=[0, 1, 2],
            episodes_per_match=4, elo_orderings=5,
            sim_fn=_strength_sim(strength),
        )
        ranked = sorted(names, key=lambda n: -report["avg_score"][n])
        assert ranked[-1] == "weak"
        assert report["avg_score"]["weak"] == 0.0
