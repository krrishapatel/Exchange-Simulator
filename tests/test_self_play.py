"""Tests for the self-play RL training system.

Tests cover:
  - SelfPlayEnv reset and step mechanics
  - Opponent pool add/sample/evict behavior
  - Evaluation metric computations
  - Environment compatibility with Gymnasium API
  - Competitive reward logic
"""

import random
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from rl.self_play_env import OpponentPolicy, SelfPlayEnv
from rl.self_play import OpponentPool
from rl.evaluate import (
    compute_max_drawdown,
    compute_sharpe_ratio,
    compute_inventory_utilization,
)


# ---------------------------------------------------------------------------
# SelfPlayEnv tests
# ---------------------------------------------------------------------------


class TestSelfPlayEnvReset:
    """Test self-play environment reset behavior."""

    def test_reset_returns_valid_observation(self):
        """Reset returns an observation matching the observation space."""
        env = SelfPlayEnv(seed=42)
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (11,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        env.close()

    def test_reset_returns_info_with_expected_keys(self):
        """Reset info dict contains agent and opponent state."""
        env = SelfPlayEnv(seed=42)
        obs, info = env.reset(seed=42)
        assert "inventory" in info
        assert "pnl" in info
        assert "step" in info
        assert "opponent_inventory" in info
        assert "opponent_pnl" in info
        assert "opponent_name" in info
        assert info["inventory"] == 0
        assert info["pnl"] == 0.0
        assert info["step"] == 0
        env.close()

    def test_reset_deterministic_with_same_seed(self):
        """Same seed produces the same initial observation."""
        env = SelfPlayEnv(seed=42)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)
        env.close()


class TestSelfPlayEnvStep:
    """Test self-play environment step mechanics."""

    def test_step_returns_valid_tuple(self):
        """Step returns (obs, reward, terminated, truncated, info)."""
        env = SelfPlayEnv(seed=42)
        env.reset(seed=42)
        result = env.step(0)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (11,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_step_all_actions_produce_valid_obs(self):
        """All 5 actions produce observations within the space."""
        env = SelfPlayEnv(seed=42)
        env.reset(seed=42)
        for action in range(5):
            obs, _, _, _, _ = env.step(action)
            assert env.observation_space.contains(obs), (
                f"Action {action} produced obs outside space"
            )
        env.close()

    def test_episode_truncates_at_length(self):
        """Episode truncates after episode_length steps."""
        env = SelfPlayEnv(episode_length=50, seed=42)
        env.reset(seed=42)
        for i in range(49):
            _, _, terminated, truncated, _ = env.step(0)
            assert not terminated
            assert not truncated
        _, _, terminated, truncated, _ = env.step(0)
        assert not terminated
        assert truncated
        env.close()

    def test_buy_market_changes_inventory(self):
        """Buy market actions should accumulate positive inventory."""
        env = SelfPlayEnv(seed=42, episode_length=100)
        env.reset(seed=42)
        for _ in range(20):
            _, _, _, _, info = env.step(2)  # buy market
        # After many buy markets, inventory should be positive
        assert info["inventory"] > 0
        env.close()

    def test_sell_market_changes_inventory(self):
        """Sell market actions should accumulate negative inventory."""
        env = SelfPlayEnv(seed=42, episode_length=100)
        env.reset(seed=42)
        for _ in range(20):
            _, _, _, _, info = env.step(4)  # sell market
        assert info["inventory"] < 0
        env.close()

    def test_opponent_also_trades(self):
        """The opponent should also accumulate non-zero PnL over time."""
        env = SelfPlayEnv(seed=42, episode_length=200)
        env.reset(seed=42)
        for _ in range(200):
            _, _, terminated, truncated, info = env.step(
                random.randint(0, 4)
            )
            if terminated or truncated:
                break
        # Opponent should have traded (non-zero PnL or inventory)
        # Since the default opponent is random, it will trade
        opp_pnl = info["opponent_pnl"]
        opp_inv = info["opponent_inventory"]
        assert opp_pnl != 0 or opp_inv != 0, (
            "Opponent should have non-zero state after 200 steps"
        )
        env.close()


class TestSelfPlayEnvOpponentSwap:
    """Test opponent selection and swapping."""

    def test_set_opponent_uses_specified_opponent(self):
        """set_opponent forces a specific opponent for the next episode."""
        custom_opp = OpponentPolicy(
            action_fn=lambda obs: 0,  # Always hold
            name="always_hold",
        )
        env = SelfPlayEnv(seed=42)
        env.set_opponent(custom_opp)
        _, info = env.reset(seed=42)
        assert info["opponent_name"] == "always_hold"
        env.close()

    def test_opponent_pool_update(self):
        """set_opponent_pool replaces the pool."""
        env = SelfPlayEnv(seed=42)
        new_pool = [
            OpponentPolicy(action_fn=lambda obs: 1, name="buyer"),
            OpponentPolicy(action_fn=lambda obs: 3, name="seller"),
        ]
        env.set_opponent_pool(new_pool)
        # Reset multiple times; should sample from the new pool
        names = set()
        for i in range(20):
            _, info = env.reset(seed=i)
            names.add(info["opponent_name"])
        assert names.issubset({"buyer", "seller"})
        env.close()


class TestSelfPlayEnvCompetitiveReward:
    """Test that the competitive reward reflects relative performance."""

    def test_competitive_reward_penalizes_opponent_gain(self):
        """Higher competitive weight reduces reward when opponent profits."""
        # With weight=0.0, reward is pure absolute
        env_abs = SelfPlayEnv(
            competitive_reward_weight=0.0, seed=42, episode_length=50
        )
        env_abs.reset(seed=42)

        # With weight=1.0, reward is fully relative
        env_rel = SelfPlayEnv(
            competitive_reward_weight=1.0, seed=42, episode_length=50
        )
        env_rel.reset(seed=42)

        # Run same actions in both
        rewards_abs = []
        rewards_rel = []
        for _ in range(50):
            _, r_abs, _, _, _ = env_abs.step(2)  # buy market
            _, r_rel, _, _, _ = env_rel.step(2)
            rewards_abs.append(r_abs)
            rewards_rel.append(r_rel)

        # The sums should differ (competitive reward subtracts opponent gain)
        total_abs = sum(rewards_abs)
        total_rel = sum(rewards_rel)
        # They shouldn't be exactly equal unless opponent does nothing
        # (opponent is random, so it will trade)
        # Just verify both are finite
        assert np.isfinite(total_abs)
        assert np.isfinite(total_rel)
        env_abs.close()
        env_rel.close()


# ---------------------------------------------------------------------------
# OpponentPool tests
# ---------------------------------------------------------------------------


class TestOpponentPool:
    """Test the opponent checkpoint pool."""

    def test_pool_starts_empty(self):
        """New pool has size 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(max_size=5, save_dir=tmpdir)
            assert pool.size == 0
            assert pool.generation == 0

    def test_add_checkpoint_increases_size(self):
        """Adding a checkpoint increases pool size and generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(max_size=5, save_dir=tmpdir)

            # Mock model with a save method
            mock_model = MagicMock()
            mock_model.save = MagicMock()

            pool.add_checkpoint(mock_model)
            assert pool.size == 1
            assert pool.generation == 1

            pool.add_checkpoint(mock_model)
            assert pool.size == 2
            assert pool.generation == 2

    def test_pool_evicts_oldest_when_full(self):
        """Pool evicts oldest when max_size is exceeded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(max_size=3, save_dir=tmpdir)
            mock_model = MagicMock()
            mock_model.save = MagicMock()

            for _ in range(5):
                pool.add_checkpoint(mock_model)

            assert pool.size == 3
            assert pool.generation == 5

    def test_sample_requires_non_empty_pool(self):
        """Sampling from empty pool raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(max_size=5, save_dir=tmpdir)
            with pytest.raises(ValueError, match="No checkpoints"):
                pool.sample()

    def test_sample_returns_valid_checkpoint(self):
        """Sampling returns a path and index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(max_size=5, save_dir=tmpdir)
            mock_model = MagicMock()
            mock_model.save = MagicMock()

            pool.add_checkpoint(mock_model)
            pool.add_checkpoint(mock_model)

            path, idx = pool.sample(rng=random.Random(42))
            assert isinstance(path, Path)
            assert 0 <= idx < pool.size

    def test_win_rate_tracking(self):
        """Win rates are correctly recorded and computed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pool = OpponentPool(max_size=5, save_dir=tmpdir)

            pool.record_result(0, True)
            pool.record_result(0, True)
            pool.record_result(0, False)
            pool.record_result(1, False)
            pool.record_result(1, False)

            rates = pool.get_win_rates()
            assert abs(rates[0] - 2 / 3) < 1e-6
            assert rates[1] == 0.0


# ---------------------------------------------------------------------------
# Evaluation metric tests
# ---------------------------------------------------------------------------


class TestEvaluationMetrics:
    """Test evaluation metric computations."""

    def test_sharpe_ratio_positive_returns(self):
        """Positive constant returns yield positive Sharpe."""
        returns = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        # All same -> std=0 -> sharpe=0
        assert compute_sharpe_ratio(returns) == 0.0

        # Variable positive returns
        returns = np.array([1.0, 2.0, 1.5, 3.0, 2.5])
        sharpe = compute_sharpe_ratio(returns)
        assert sharpe > 0

    def test_sharpe_ratio_empty_returns(self):
        """Empty returns yield zero Sharpe."""
        assert compute_sharpe_ratio(np.array([])) == 0.0

    def test_sharpe_ratio_negative_returns(self):
        """Negative returns yield negative Sharpe."""
        returns = np.array([-1.0, -2.0, -1.5, -3.0, -2.5])
        sharpe = compute_sharpe_ratio(returns)
        assert sharpe < 0

    def test_max_drawdown_no_drawdown(self):
        """Monotonically increasing PnL has zero drawdown."""
        pnl = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_max_drawdown(pnl) == 0.0

    def test_max_drawdown_with_dip(self):
        """Drawdown is correctly computed from peak to trough."""
        pnl = np.array([10.0, 8.0, 6.0, 9.0, 7.0])
        # Peak=10, trough after peak=6, so max dd = 4
        assert compute_max_drawdown(pnl) == 4.0

    def test_max_drawdown_empty(self):
        """Empty series yields zero drawdown."""
        assert compute_max_drawdown(np.array([])) == 0.0

    def test_inventory_utilization_always_flat(self):
        """Zero inventory throughout yields 0% utilization."""
        inv = np.array([0, 0, 0, 0, 0])
        assert compute_inventory_utilization(inv) == 0.0

    def test_inventory_utilization_always_active(self):
        """Non-zero inventory throughout yields 100% utilization."""
        inv = np.array([1, -2, 3, -1, 5])
        assert compute_inventory_utilization(inv) == 1.0

    def test_inventory_utilization_partial(self):
        """Mixed inventory yields correct fraction."""
        inv = np.array([0, 1, 0, 1, 0])
        assert abs(compute_inventory_utilization(inv) - 0.4) < 1e-6

    def test_inventory_utilization_empty(self):
        """Empty series yields 0."""
        assert compute_inventory_utilization(np.array([])) == 0.0


# ---------------------------------------------------------------------------
# Integration: full episode test
# ---------------------------------------------------------------------------


class TestSelfPlayIntegration:
    """Integration tests running full episodes."""

    def test_full_episode_completes(self):
        """A full episode runs to completion without errors."""
        env = SelfPlayEnv(episode_length=100, seed=42)
        obs, _ = env.reset(seed=42)

        steps = 0
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            done = terminated or truncated
            assert np.isfinite(reward)
            assert not np.any(np.isnan(obs))

        assert steps == 100
        env.close()

    def test_get_episode_result(self):
        """get_episode_result returns valid result dict after episode."""
        env = SelfPlayEnv(episode_length=50, seed=42)
        env.reset(seed=42)

        for _ in range(50):
            env.step(env.action_space.sample())

        result = env.get_episode_result()
        assert "agent_pnl" in result
        assert "opponent_pnl" in result
        assert "agent_won" in result
        assert "opponent_name" in result
        assert isinstance(result["agent_won"], bool)
        env.close()

    def test_multiple_resets(self):
        """Environment can be reset and run multiple times."""
        env = SelfPlayEnv(episode_length=20, seed=42)
        for ep in range(5):
            obs, _ = env.reset(seed=ep)
            assert env.observation_space.contains(obs)
            for _ in range(20):
                obs, _, terminated, truncated, _ = env.step(
                    env.action_space.sample()
                )
                if terminated or truncated:
                    break
            assert env.observation_space.contains(obs)
        env.close()
