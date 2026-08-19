"""Tests for order expiry in RandomAgent and requoting in the RL envs.

Both exist because resting orders that are never cancelled destroy the market.
Three RandomAgents add about 16 units of size per step and only ~8% of flow
crosses, so without expiry the book thickens without bound: over 3000 steps the
touch grew to 3258 units and the mid took 13 distinct values. Every measurement
in this repo is taken against that book, so these tests guard the thing the rest
of the numbers rest on.
"""

import statistics

import pytest

import exchange_simulator as ex
from agents.random_agent import RandomAgent
from rl.trading_env import TradingEnv


def run_agents(agents, steps, engine=None):
    """Drive agents against one engine, routing fills back to them."""
    engine = engine or ex.MatchingEngine()
    owner = {}
    for step in range(steps):
        timestamp = step * 1_000_000
        for agent in agents:
            for order in agent.on_market_data(engine, timestamp):
                owner[order.id] = agent
                for fill in engine.submit(order):
                    for order_id in (fill.maker_order_id, fill.taker_order_id):
                        if order_id in owner:
                            owner[order_id].on_fill(fill)
    return engine


def total_resting(book):
    """Units and orders resting on both sides."""
    bids = book.l2_bids(10_000)
    asks = book.l2_asks(10_000)
    return (
        sum(level.total_quantity for level in bids)
        + sum(level.total_quantity for level in asks),
        sum(level.order_count for level in bids)
        + sum(level.order_count for level in asks),
    )


class TestRandomAgentExpiry:
    """Resting orders have to leave the book on their own."""

    def test_depth_is_stationary_with_expiry(self):
        """The book should not be much deeper at step 2000 than at step 1000."""
        agents = [RandomAgent(agent_id=i, seed=100 + i) for i in range(1, 4)]
        engine = run_agents(agents, 1000)
        early_units, _ = total_resting(engine.book())
        run_agents(agents, 1000, engine=engine)
        late_units, _ = total_resting(engine.book())

        assert late_units < early_units * 2, (
            f"resting size went {early_units} -> {late_units}, which is growth "
            "rather than a steady state"
        )

    def test_depth_grows_without_expiry(self):
        """order_lifetime=0 reproduces the runaway book, so the test above has teeth."""
        agents = [
            RandomAgent(agent_id=i, seed=100 + i, order_lifetime=0) for i in range(1, 4)
        ]
        engine = run_agents(agents, 1000)
        early_units, _ = total_resting(engine.book())
        run_agents(agents, 1000, engine=engine)
        late_units, _ = total_resting(engine.book())

        assert late_units > early_units * 1.5, (
            f"expected the book to keep growing without expiry, got "
            f"{early_units} -> {late_units}"
        )

    def test_expiry_keeps_the_price_moving(self):
        """The mid has to visit many prices. Without expiry it froze on 13."""
        agents = [RandomAgent(agent_id=i, seed=7 + i) for i in range(1, 4)]
        engine = ex.MatchingEngine()
        owner = {}
        mids = []
        for step in range(2000):
            for agent in agents:
                for order in agent.on_market_data(engine, step * 1_000_000):
                    owner[order.id] = agent
                    for fill in engine.submit(order):
                        for order_id in (fill.maker_order_id, fill.taker_order_id):
                            if order_id in owner:
                                owner[order_id].on_fill(fill)
            book = engine.book()
            bid, ask = book.best_bid_price(), book.best_ask_price()
            if bid is not None and ask is not None:
                mids.append((bid + ask) / 2.0)

        distinct = len(set(mids))
        moves = sum(1 for a, b in zip(mids, mids[1:]) if a != b) / len(mids)
        assert distinct > 30, f"mid only took {distinct} distinct values"
        assert moves > 0.05, f"mid moved on {moves:.1%} of steps"

    def test_orders_are_cancelled_after_their_lifetime(self):
        """An order placed early must be gone once its lifetime has passed."""
        agent = RandomAgent(agent_id=1, seed=5, order_lifetime=10, aggression=0.0)
        engine = ex.MatchingEngine()

        submitted = []
        for step in range(3):
            for order in agent.on_market_data(engine, step * 1_000_000):
                engine.submit(order)
                submitted.append(order.id)

        _, orders_before = total_resting(engine.book())
        assert orders_before > 0, "nothing rested, the test cannot show anything"

        # Run past the lifetime. Aggression is off, so nothing here crosses and
        # anything that leaves the book left because it was cancelled.
        for step in range(3, 40):
            for order in agent.on_market_data(engine, step * 1_000_000):
                engine.submit(order)

        # The engine exposes no order-id listing, so check the count instead: with
        # a lifetime of 10 and one order per step, the book holds about 10 orders
        # rather than the 40 submitted.
        _, orders_after = total_resting(engine.book())
        assert orders_after <= 12, (
            f"{orders_after} orders resting after 40 steps at lifetime 10, "
            "so expiry is not firing"
        )

    def test_lifetime_zero_keeps_everything(self):
        """The escape hatch has to actually disable expiry."""
        agent = RandomAgent(agent_id=1, seed=5, order_lifetime=0, aggression=0.0)
        engine = ex.MatchingEngine()
        for step in range(30):
            for order in agent.on_market_data(engine, step * 1_000_000):
                engine.submit(order)
        _, orders = total_resting(engine.book())
        assert orders >= 28, f"only {orders} of ~30 orders still resting"


class TestTradingEnvRequoting:
    """The RL agent's own quotes must not pile up in the book it is marked against."""

    @pytest.mark.parametrize("action", [1, 3])
    def test_quotes_do_not_accumulate(self, action):
        """Quoting every step must leave the book as it would have been on hold.

        Before requoting, 1000 steps of action 1 grew the bid side from 375 units
        in 67 orders to 746 in 438, so the agent became half the liquidity it was
        being valued against.
        """
        baseline = TradingEnv(episode_length=400, seed=11)
        baseline.reset(seed=11)
        for _ in range(400):
            baseline.step(0)
        idle_units, idle_orders = total_resting(baseline._engine.book())

        env = TradingEnv(episode_length=400, seed=11)
        env.reset(seed=11)
        for _ in range(400):
            env.step(action)
        units, orders = total_resting(env._engine.book())

        assert orders <= idle_orders + 2, (
            f"action {action} left {orders} orders resting against {idle_orders} "
            "when holding, so old quotes are not being cancelled"
        )
        assert units <= idle_units + 20

    def test_market_orders_are_not_tracked_for_cancellation(self):
        """IOC orders never rest, so nothing should be queued to cancel."""
        env = TradingEnv(episode_length=50, seed=3)
        env.reset(seed=3)
        for _ in range(50):
            env.step(2)  # buy market
        assert env._live_order_ids == []


class TestTradingEnvPositionLimit:
    """An unbounded position marked at the mid is not a result."""

    @pytest.mark.parametrize("action", [1, 2, 3, 4])
    def test_limit_is_never_breached(self, action):
        env = TradingEnv(episode_length=600, max_inventory=8, seed=21)
        env.reset(seed=21)
        for _ in range(600):
            _, _, _, _, info = env.step(action)
            assert abs(info["inventory"]) <= 8, (
                f"action {action} reached inventory {info['inventory']} "
                "against a limit of 8"
            )

    def test_limit_binds(self):
        """Buying every step should reach the limit, or the test above proves nothing."""
        env = TradingEnv(episode_length=600, max_inventory=8, seed=21)
        env.reset(seed=21)
        peak = 0
        for _ in range(600):
            _, _, _, _, info = env.step(2)
            peak = max(peak, info["inventory"])
        assert peak == 8, f"only reached {peak} of a limit of 8"

    def test_zero_disables_the_limit(self):
        """max_inventory=0 is the documented escape hatch for showing the old behaviour."""
        env = TradingEnv(episode_length=600, max_inventory=0, seed=21)
        env.reset(seed=21)
        for _ in range(600):
            _, _, _, _, info = env.step(2)
        assert info["inventory"] > 50, (
            f"expected an unbounded position, got {info['inventory']}"
        )

    def test_blocked_action_still_expires_the_old_quote(self):
        """Hitting the limit must not leave a stale quote resting forever."""
        env = TradingEnv(episode_length=200, max_inventory=3, seed=21)
        env.reset(seed=21)
        for _ in range(200):
            env.step(1)  # buy limit, blocked once the limit is reached
        assert len(env._live_order_ids) <= 1


class TestObservationIsReal:
    """The depth features have to describe the book, not a made-up split of a count."""

    def test_level_quantities_match_the_book(self):
        env = TradingEnv(episode_length=100, seed=31)
        obs, _ = env.reset(seed=31)
        for _ in range(100):
            obs, _, _, _, _ = env.step(0)

        book = env._engine.book()
        bids = book.l2_bids(3)
        asks = book.l2_asks(3)
        scale = TradingEnv._LEVEL_QTY_SCALE

        for i, level in enumerate(bids):
            assert obs[2 + i] == pytest.approx(level.total_quantity / scale, rel=1e-5)
        for i, level in enumerate(asks):
            assert obs[5 + i] == pytest.approx(level.total_quantity / scale, rel=1e-5)

    def test_levels_are_not_a_fixed_ratio(self):
        """The old code split one number 3:2:1, so the levels were never independent."""
        env = TradingEnv(episode_length=300, seed=32)
        obs, _ = env.reset(seed=32)
        ratios = []
        for _ in range(300):
            obs, _, _, _, _ = env.step(0)
            if obs[2] > 0:
                ratios.append(obs[3] / obs[2])
        assert len(set(round(r, 3) for r in ratios)) > 5, (
            "level 1 is a constant multiple of level 0, so it carries nothing new"
        )

    def test_imbalance_is_quantity_not_level_count(self):
        env = TradingEnv(episode_length=100, seed=33)
        obs, _ = env.reset(seed=33)
        for _ in range(100):
            obs, _, _, _, _ = env.step(0)

        book = env._engine.book()
        bid_qty = book.l2_bids(1)[0].total_quantity
        ask_qty = book.l2_asks(1)[0].total_quantity
        expected = (bid_qty - ask_qty) / (bid_qty + ask_qty)
        assert obs[10] == pytest.approx(expected, rel=1e-5)

    def test_inventory_feature_is_scaled_by_the_limit(self):
        env = TradingEnv(episode_length=100, max_inventory=10, seed=34)
        env.reset(seed=34)
        for _ in range(100):
            obs, _, _, _, info = env.step(2)
        assert obs[8] == pytest.approx(info["inventory"] / 10.0, rel=1e-5)
        assert -1.0 <= obs[8] <= 1.0

    def test_equity_feature_is_equity_not_cash(self):
        """obs[9] used to be cash, which reads as a huge loss for anyone holding stock."""
        env = TradingEnv(episode_length=100, seed=35)
        env.reset(seed=35)
        for _ in range(100):
            obs, _, _, _, info = env.step(2)  # buy, so cash and equity diverge
        assert info["inventory"] > 0, "need a position for the two to differ"
        assert obs[9] == pytest.approx(
            info["equity"] / TradingEnv._EQUITY_SCALE, rel=1e-5
        )
        assert obs[9] > info["pnl"] / TradingEnv._EQUITY_SCALE

    def test_features_stay_on_a_sane_scale(self):
        """Every feature should be order 1, so no input dominates by units alone."""
        env = TradingEnv(episode_length=500, seed=36)
        obs, _ = env.reset(seed=36)
        seen = []
        for i in range(500):
            obs, _, _, _, _ = env.step(1 if i % 2 else 3)
            seen.append(obs)

        for index in range(11):
            worst = max(abs(row[index]) for row in seen)
            assert worst < 20.0, f"feature {index} reached {worst:.1f}"


class TestSelfPlayEnvParity:
    """Both envs have to present the same observation, or a checkpoint cannot move."""

    def test_same_observation_layout(self):
        from rl.self_play_env import SelfPlayEnv

        trading = TradingEnv(seed=42)
        self_play = SelfPlayEnv(seed=42)
        assert trading.observation_space.shape == self_play.observation_space.shape
        for name in ("_MID_SCALE", "_SPREAD_SCALE", "_LEVEL_QTY_SCALE", "_EQUITY_SCALE"):
            assert getattr(trading, name) == getattr(self_play, name), name

    def test_self_play_respects_the_position_limit(self):
        from rl.self_play_env import SelfPlayEnv

        env = SelfPlayEnv(episode_length=400, max_inventory=6, seed=9)
        env.reset(seed=9)
        for _ in range(400):
            _, _, _, _, info = env.step(2)
            assert abs(info["inventory"]) <= 6
            assert abs(info["opponent_inventory"]) <= 6

    def test_self_play_quotes_do_not_accumulate(self):
        from rl.self_play_env import SelfPlayEnv

        idle = SelfPlayEnv(episode_length=300, seed=9)
        idle.reset(seed=9)
        for _ in range(300):
            idle.step(0)
        _, idle_orders = total_resting(idle._engine.book())

        env = SelfPlayEnv(episode_length=300, seed=9)
        env.reset(seed=9)
        for _ in range(300):
            env.step(1)
        _, orders = total_resting(env._engine.book())
        # The opponent quotes too, so allow its single live order on top.
        assert orders <= idle_orders + 4


class TestBaselines:
    """The reference policies, since a learning curve alone proves nothing."""

    def test_hold_scores_exactly_zero(self):
        from rl.baselines import Hold, run_baseline

        env = TradingEnv(episode_length=300, seed=51)
        result = run_baseline(Hold(), env, seed=51)
        assert result["equity"] == 0.0
        assert result["max_abs_inventory"] == 0

    def test_random_loses_money(self):
        """Two of five actions cross the spread, so acting at random must cost."""
        from rl.baselines import Random, run_baseline

        env = TradingEnv(episode_length=1000, seed=52)
        equities = [run_baseline(Random(seed=s), env, seed=s) for s in range(3)]
        assert statistics.mean(r["equity"] for r in equities) < -100

    def test_inventory_aware_makes_money_while_staying_flat(self):
        """The bar the trained agent has to clear."""
        from rl.baselines import InventoryAware, run_baseline

        env = TradingEnv(episode_length=1000, seed=53)
        results = [run_baseline(InventoryAware(), env, seed=s) for s in range(1, 11)]
        mean_equity = statistics.mean(r["equity"] for r in results)
        assert mean_equity > 5, f"expected a positive edge, got {mean_equity:.1f}"
        assert max(r["max_abs_inventory"] for r in results) <= 2, (
            "this policy is supposed to earn the spread without taking a position"
        )

    def test_every_baseline_runs(self):
        from rl.baselines import all_baselines, run_baseline

        env = TradingEnv(episode_length=200, seed=54)
        for policy in all_baselines():
            result = run_baseline(policy, env, seed=54)
            assert result["policy"] == policy.name
            assert "equity" in result

    def test_baselines_are_reproducible(self):
        from rl.baselines import all_baselines, run_baseline

        env = TradingEnv(episode_length=200, seed=55)
        first = [run_baseline(p, env, seed=55) for p in all_baselines()]
        second = [run_baseline(p, env, seed=55) for p in all_baselines()]
        assert [r["equity"] for r in first] == [r["equity"] for r in second]
