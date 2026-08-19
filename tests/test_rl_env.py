"""Tests for the RL trading environment."""

import numpy as np

import gymnasium
from gymnasium.utils.env_checker import check_env

from rl import TradingEnv


class TestTradingEnvCreation:
    """Test environment creation and configuration."""

    def test_env_creates(self):
        """Environment can be instantiated."""
        env = TradingEnv(seed=42)
        assert env is not None
        env.close()

    def test_observation_space(self):
        """Observation space is Box with correct shape."""
        env = TradingEnv(seed=42)
        assert isinstance(env.observation_space, gymnasium.spaces.Box)
        assert env.observation_space.shape == (11,)
        assert env.observation_space.dtype == np.float32
        env.close()

    def test_action_space(self):
        """Action space is Discrete(5)."""
        env = TradingEnv(seed=42)
        assert isinstance(env.action_space, gymnasium.spaces.Discrete)
        assert env.action_space.n == 5
        env.close()


class TestTradingEnvReset:
    """Test environment reset behavior."""

    def test_reset_returns_valid_obs(self):
        """Reset returns observation in the observation space."""
        env = TradingEnv(seed=42)
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (11,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        env.close()

    def test_reset_returns_info_dict(self):
        """Reset returns an info dictionary with expected keys."""
        env = TradingEnv(seed=42)
        obs, info = env.reset(seed=42)
        assert isinstance(info, dict)
        assert "inventory" in info
        assert "pnl" in info
        assert "step" in info
        assert info["inventory"] == 0
        assert info["pnl"] == 0.0
        assert info["step"] == 0
        env.close()

    def test_reset_deterministic_with_seed(self):
        """Same seed produces same initial observation."""
        env = TradingEnv(seed=42)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)
        env.close()


class TestTradingEnvStep:
    """Test environment step behavior."""

    def test_step_returns_valid_tuple(self):
        """Step returns (obs, reward, terminated, truncated, info)."""
        env = TradingEnv(seed=42)
        env.reset(seed=42)
        result = env.step(0)  # hold action
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (11,)
        assert obs.dtype == np.float32
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_step_obs_in_space(self):
        """Step observation is within the observation space."""
        env = TradingEnv(seed=42)
        env.reset(seed=42)
        for action in range(5):
            obs, _, _, _, _ = env.step(action)
            assert env.observation_space.contains(obs), (
                f"Action {action} produced obs outside space: {obs}"
            )
        env.close()

    def test_hold_action_no_inventory_change(self):
        """Hold action should not change inventory."""
        env = TradingEnv(seed=42)
        env.reset(seed=42)
        _, _, _, _, info = env.step(0)
        assert info["inventory"] == 0
        env.close()

    def test_buy_market_increases_inventory(self):
        """Buy market action should increase inventory (when book has asks)."""
        env = TradingEnv(seed=42)
        env.reset(seed=42)
        # Execute buy market multiple times to ensure at least one fill
        total_inv = 0
        for _ in range(10):
            _, _, _, _, info = env.step(2)  # buy_market
            total_inv = info["inventory"]
        # With noise traders providing liquidity, we should get some fills
        assert total_inv > 0, "Expected positive inventory after buy market orders"
        env.close()

    def test_sell_market_decreases_inventory(self):
        """Sell market action should decrease inventory (when book has bids)."""
        env = TradingEnv(seed=42)
        env.reset(seed=42)
        total_inv = 0
        for _ in range(10):
            _, _, _, _, info = env.step(4)  # sell_market
            total_inv = info["inventory"]
        assert total_inv < 0, "Expected negative inventory after sell market orders"
        env.close()


class TestTradingEnvEpisode:
    """Test full episode behavior."""

    def test_episode_runs_to_completion(self):
        """Episode completes after episode_length steps."""
        env = TradingEnv(episode_length=1000, seed=42)
        env.reset(seed=42)

        step_count = 0
        done = False
        while not done:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            step_count += 1
            done = terminated or truncated

        assert step_count == 1000
        env.close()

    def test_episode_truncated_not_terminated(self):
        """Episode ends via truncation, not termination."""
        env = TradingEnv(episode_length=100, seed=42)
        env.reset(seed=42)

        terminated = False
        truncated = False
        for _ in range(100):
            _, _, terminated, truncated, _ = env.step(0)

        assert not terminated
        assert truncated
        env.close()

    def test_short_episode(self):
        """Short episodes work correctly."""
        env = TradingEnv(episode_length=10, seed=42)
        env.reset(seed=42)

        for i in range(9):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            assert not terminated
            assert not truncated

        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert truncated
        env.close()


class TestTradingEnvGymCompliance:
    """Test Gymnasium API compliance."""

    def test_check_env(self):
        """Environment passes gymnasium's built-in env_checker."""
        env = TradingEnv(episode_length=100, seed=42)
        # check_env raises an exception or prints warnings if non-compliant
        check_env(env.unwrapped, skip_render_check=True)
        env.close()

    def test_multiple_resets(self):
        """Environment can be reset multiple times."""
        env = TradingEnv(seed=42)
        for i in range(5):
            obs, info = env.reset(seed=i)
            assert env.observation_space.contains(obs)
            # Take a few steps
            for _ in range(10):
                obs, _, _, _, _ = env.step(env.action_space.sample())
                assert env.observation_space.contains(obs)
        env.close()

    def test_no_nan_in_observations(self):
        """Observations should never contain NaN values."""
        env = TradingEnv(episode_length=200, seed=42)
        env.reset(seed=42)
        for _ in range(200):
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            assert not np.any(np.isnan(obs)), f"NaN found in observation: {obs}"
            if terminated or truncated:
                break
        env.close()


class TestTradingEnvReward:
    """The reward has to track profit, not cash.

    `_agent_pnl` is cash flow: a buy subtracts the whole notional. Rewarding its
    change directly makes selling look good and buying look catastrophic no
    matter what the prices were. Measured over 500 steps before this was fixed,
    always-buy-market scored -500,005,283 while being the most profitable policy
    at +31,592 marked, and always-sell-market scored +499,927,312 for half the
    profit. These tests pin the reward to marked equity so that cannot come back.
    """

    PENALTY = 0.001

    def _roll(self, action, steps=300, seed=7):
        """Run one fixed-action episode, returning reward and equity totals."""
        env = TradingEnv(
            episode_length=steps, inventory_penalty=self.PENALTY, seed=seed
        )
        env.reset(seed=seed)
        total_reward = 0.0
        total_penalty = 0.0
        info = {}
        for _ in range(steps):
            _, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_penalty += self.PENALTY * abs(info["inventory"])
            if terminated or truncated:
                break
        return total_reward, total_penalty, info

    def test_reward_sums_to_marked_equity(self):
        """Rewards telescope to final marked equity, less the penalties charged.

        On the cash-based reward this sum came out at the cash balance instead,
        which for a buying policy is hundreds of millions away from the truth.
        """
        for action in range(5):
            total_reward, total_penalty, info = self._roll(action)
            expected = info["equity"]
            assert abs(total_reward + total_penalty - expected) < 1.0, (
                f"action {action}: rewards sum to "
                f"{total_reward + total_penalty:.0f} but equity is {expected:.0f}"
            )

    def test_buying_is_not_punished_for_being_a_purchase(self):
        """Buying at the touch costs the spread, not the notional."""
        total_reward, _, info = self._roll(2)  # buy market every step
        assert info["inventory"] > 0, "buy_market should end long"
        # The old reward was about -1e6 per share bought.
        assert total_reward > -10 * info["inventory"], (
            f"reward {total_reward:.0f} for {info['inventory']} shares looks like "
            "the notional is being charged rather than the spread"
        )

    def test_reward_ranks_profitable_policies_above_losing_ones(self):
        """Every policy that made money must score above every one that lost it.

        Not a full ordering: the inventory penalty is part of the objective, so
        two policies within a penalty's width of each other can legitimately
        swap. What must never happen is the old behaviour, where the losing
        policies scored half a billion above the winning ones.
        """
        scored = []
        for action in range(5):
            total_reward, _, info = self._roll(action)
            scored.append((action, total_reward, info["equity"]))

        winners = [row for row in scored if row[2] > 0]
        losers = [row for row in scored if row[2] < 0]
        assert winners and losers, f"need both to compare, got {scored}"
        assert min(r for _, r, _ in winners) > max(r for _, r, _ in losers), (
            f"reward puts a losing policy above a profitable one: {scored}"
        )
