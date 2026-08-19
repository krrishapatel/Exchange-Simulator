"""Tests for the round-robin tournament runner.

This file exists because the runner had no tests, and so nobody noticed that it
was scoring matches with `rng.normal(0, 100)` for both players. It printed an Elo
table and head-to-head win rates that did not depend on the agents at all, which
is worse than printing nothing. The tests below all fail against that version.

An agent that never trades is the useful probe: under a real simulation its PnL
is exactly zero and it cannot win a match, and under a fabricated one it picks up
a respectable rating for doing nothing.
"""

from rl.analysis.tournament import AgentEntry, MatchResult, Tournament

HOLD, BUY_MARKET, SELL_MARKET = 0, 2, 4


def _entry(name, action):
    """An agent that always takes the same action."""
    return AgentEntry(name=name, action_fn=lambda obs: action)


def _tournament(**kwargs):
    agents = [
        _entry("idle", HOLD),
        _entry("buyer", BUY_MARKET),
        _entry("seller", SELL_MARKET),
    ]
    params = dict(episodes_per_match=2, episode_length=100, seed=42)
    params.update(kwargs)
    return Tournament(agents, **params)


class TestMatchResult:
    """Winner and Elo score bookkeeping."""

    def test_higher_pnl_wins(self):
        result = MatchResult("a", "b", pnl_a=10.0, pnl_b=5.0)
        assert result.winner == "a"
        assert result.result_for_a == 1.0

    def test_lower_pnl_loses(self):
        result = MatchResult("a", "b", pnl_a=5.0, pnl_b=10.0)
        assert result.winner == "b"
        assert result.result_for_a == 0.0

    def test_equal_pnl_draws(self):
        result = MatchResult("a", "b", pnl_a=7.0, pnl_b=7.0)
        assert result.winner == "draw"
        assert result.result_for_a == 0.5


class TestTournamentStructure:
    """Round-robin bookkeeping, independent of how matches are simulated."""

    def test_num_matchups_is_n_choose_2(self):
        assert _tournament().num_matchups == 3

    def test_every_agent_plays_every_other(self):
        tournament = _tournament()
        results = tournament.run()
        assert results["num_matchups"] == 3
        for name in ("idle", "buyer", "seller"):
            assert results["elo_ratings"][name] is not None
            # Two opponents, two episodes each.
            assert tournament.elo.get_games_played(name) == 4

    def test_summary_table_lists_all_agents(self):
        table = _tournament().run()["summary_table"]
        for name in ("idle", "buyer", "seller"):
            assert name in table


class TestMatchesReflectTheAgents:
    """The results have to be a function of the agents playing."""

    def test_idle_agent_never_makes_or_loses_money(self):
        """An agent that only holds must score exactly zero, not merely little.

        Under the fabricated simulation this was a draw from a normal
        distribution, so it came out near +-100 and sometimes won.
        """
        results = _tournament().run()
        idle_pnls = [
            pnl for (player, _), pnl in results["avg_pnl"].items() if player == "idle"
        ]
        assert idle_pnls, "idle agent played no matches"
        assert all(pnl == 0.0 for pnl in idle_pnls), (
            f"an agent that never trades reported PnL {idle_pnls}"
        )

    def test_head_to_head_agrees_with_pnl(self):
        """Whoever made more money in a matchup must not have the losing record.

        Note there is no assertion that the idle agent comes last. Doing nothing
        legitimately beats a policy that loses money, and both of these fixed
        policies can, so ranking it last would be a flaky test rather than a
        property of the tournament.
        """
        results = _tournament().run()
        rates, pnls = results["win_rates"], results["avg_pnl"]
        for (a, b), pnl_a in pnls.items():
            pnl_b = pnls[(b, a)]
            if pnl_a > pnl_b:
                assert rates[(a, b)] >= 0.5, (
                    f"{a} out-earned {b} ({pnl_a:.0f} vs {pnl_b:.0f}) "
                    f"but won only {rates[(a, b)]:.0%}"
                )

    def test_trading_agents_actually_trade(self):
        """Both non-idle agents move money, in at least one direction."""
        results = _tournament().run()
        for player in ("buyer", "seller"):
            pnls = [
                pnl for (a, _), pnl in results["avg_pnl"].items() if a == player
            ]
            assert any(pnl != 0.0 for pnl in pnls), f"{player} never traded"

    def test_same_seed_reproduces_results(self):
        first = _tournament().run()["avg_pnl"]
        second = _tournament().run()["avg_pnl"]
        assert first == second

    def test_swapping_a_policy_changes_the_outcome(self):
        """Results depend on the policies, not just on the seed.

        The fabricated simulation keyed only off the seed, so this was equal.
        """
        baseline = _tournament().run()["avg_pnl"]
        swapped = Tournament(
            [_entry("idle", HOLD), _entry("buyer", BUY_MARKET), _entry("seller", HOLD)],
            episodes_per_match=2,
            episode_length=100,
            seed=42,
        ).run()["avg_pnl"]
        assert baseline[("buyer", "seller")] != swapped[("buyer", "seller")]


class TestSyntheticSimulation:
    """The fabricated scorer is still available, but only on request."""

    def test_not_used_by_default(self):
        """The default must be the real simulation.

        Checked through behaviour rather than identity: the idle agent scores a
        clean zero under a real match and does not under a random draw.
        """
        tournament = _tournament()
        results = tournament.run()
        idle_pnls = [
            pnl for (player, _), pnl in results["avg_pnl"].items() if player == "idle"
        ]
        assert all(pnl == 0.0 for pnl in idle_pnls)

    def test_available_explicitly(self):
        tournament = _tournament()
        results = tournament.run(sim_fn=tournament._synthetic_simulate)
        assert results["num_matchups"] == 3

    def test_ignores_the_agents_as_documented(self):
        """Kept as a warning: this is what the default used to do."""
        tournament = _tournament()
        results = tournament.run(sim_fn=tournament._synthetic_simulate)
        idle_pnls = [
            pnl for (player, _), pnl in results["avg_pnl"].items() if player == "idle"
        ]
        assert any(pnl != 0.0 for pnl in idle_pnls), (
            "the synthetic scorer is supposed to fabricate PnL for everyone"
        )
