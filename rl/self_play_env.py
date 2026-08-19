"""Self-play Gymnasium environment for league-style training.

Implements a single-agent Gymnasium env where opponents are frozen copies
of previous policy checkpoints (league training). The RL agent trains
against a pool of its own past selves plus noise traders.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import exchange_simulator as ex
from agents.random_agent import RandomAgent


class OpponentPolicy:
    """Wraps a frozen policy checkpoint for use as an opponent.

    The opponent observes the same order book and takes actions
    according to its saved policy weights.
    """

    def __init__(
        self,
        policy_path: str | Path | None = None,
        action_fn: Callable[[np.ndarray], int] | None = None,
        name: str = "opponent",
    ):
        """Initialize an opponent policy.

        Args:
            policy_path: Path to a stable-baselines3 model zip file.
            action_fn: Alternative callable that maps observation -> action.
                       Used for random/market-maker opponents.
            name: Human-readable name for this opponent.
        """
        self.name = name
        self._model = None
        self._action_fn = action_fn

        if policy_path is not None:
            self._load_model(policy_path)

    def _load_model(self, path: str | Path) -> None:
        """Load a stable-baselines3 PPO model from disk."""
        try:
            from stable_baselines3 import PPO

            self._model = PPO.load(str(path))
        except ImportError:
            raise ImportError(
                "stable-baselines3 is required to load policy checkpoints. "
                "Install with: pip install 'stable-baselines3>=2.0'"
            )

    def predict(self, obs: np.ndarray) -> int:
        """Predict an action given an observation.

        Args:
            obs: Observation vector (shape (11,)).

        Returns:
            Integer action from {0, 1, 2, 3, 4}.
        """
        if self._action_fn is not None:
            return self._action_fn(obs)
        if self._model is not None:
            action, _ = self._model.predict(obs, deterministic=True)
            return int(action)
        # Fallback: random action
        return random.randint(0, 4)


class SelfPlayEnv(gym.Env):
    """Gymnasium environment for self-play RL training.

    A single RL agent trains against one or more opponent policies
    (frozen checkpoints of its own past selves) plus noise traders.
    The env is compatible with stable-baselines3's single-agent API.

    The self-play aspect is implemented by swapping out opponent policies
    between episodes rather than using a multi-agent API.

    Observation space (Box, float32, shape=(11,)):
        Same as TradingEnv: mid_price, spread, bid/ask depths, inventory, pnl, imbalance.

    Action space (Discrete(5)):
        Same as TradingEnv: hold, buy_limit_bid, buy_market, sell_limit_ask, sell_market.

    Reward:
        PnL delta relative to opponent's PnL delta (competitive reward)
        plus inventory penalty.
    """

    metadata = {"render_modes": []}

    _BASE_PRICE = 100_0000  # 100.0000 in fixed-point

    def __init__(
        self,
        opponent_policies: list[OpponentPolicy] | None = None,
        num_noise_traders: int = 3,
        episode_length: int = 1000,
        inventory_penalty: float = 0.001,
        order_quantity: int = 1,
        competitive_reward_weight: float = 0.5,
        seed: int | None = None,
    ):
        """Initialize the self-play environment.

        Args:
            opponent_policies: List of frozen opponent policies. If None or empty,
                               a random opponent is used.
            num_noise_traders: Number of background noise traders.
            episode_length: Steps per episode.
            inventory_penalty: Penalty multiplier on abs(inventory).
            order_quantity: Quantity for each order.
            competitive_reward_weight: Weight for relative PnL vs opponent (0-1).
                                       0 = pure absolute PnL, 1 = pure relative.
            seed: Random seed.
        """
        super().__init__()

        self.num_noise_traders = num_noise_traders
        self.episode_length = episode_length
        self.inventory_penalty = inventory_penalty
        self.order_quantity = order_quantity
        self.competitive_reward_weight = competitive_reward_weight
        self._seed = seed

        # Opponent pool
        if opponent_policies and len(opponent_policies) > 0:
            self._opponent_pool = list(opponent_policies)
        else:
            # Default: random opponent
            self._opponent_pool = [
                OpponentPolicy(
                    action_fn=lambda obs: random.randint(0, 4),
                    name="random_baseline",
                )
            ]
        self._current_opponent: OpponentPolicy | None = None

        # Observation: 11 floats (same as TradingEnv)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )

        # Actions: hold, buy_limit_at_bid, buy_market, sell_limit_at_ask, sell_market
        self.action_space = spaces.Discrete(5)

        # Internal state
        self._engine: ex.MatchingEngine | None = None
        self._noise_traders: list[RandomAgent] = []
        self._step_count = 0
        self._rng = random.Random(seed)

        # RL agent state
        self._agent_inventory = 0
        self._agent_pnl = 0.0
        self._prev_agent_pnl = 0.0
        self._agent_order_id_counter = 900_000_000
        self._my_order_ids: set[int] = set()

        # Opponent state
        self._opp_inventory = 0
        self._opp_pnl = 0.0
        self._prev_opp_pnl = 0.0
        self._opp_order_id_counter = 800_000_000
        self._opp_order_ids: set[int] = set()

        # Fill routing
        self._order_to_agent: dict[int, RandomAgent] = {}

        # Marked-equity baselines for the reward
        self._prev_agent_equity = 0.0
        self._prev_opp_equity = 0.0

        # Episode stats
        self._episode_agent_pnl = 0.0
        self._episode_opp_pnl = 0.0

    def set_opponent(self, opponent: OpponentPolicy) -> None:
        """Set a specific opponent for the next episode.

        Args:
            opponent: The OpponentPolicy to use.
        """
        self._current_opponent = opponent

    def set_opponent_pool(self, pool: list[OpponentPolicy]) -> None:
        """Replace the opponent pool.

        Args:
            pool: New list of OpponentPolicy instances.
        """
        self._opponent_pool = list(pool)

    def _sample_opponent(self) -> OpponentPolicy:
        """Sample an opponent from the pool.

        50% chance of most recent, 50% uniform over the rest.
        """
        if len(self._opponent_pool) == 1:
            return self._opponent_pool[0]

        if self._rng.random() < 0.5:
            # Most recent
            return self._opponent_pool[-1]
        else:
            # Uniform over all
            return self._rng.choice(self._opponent_pool)

    def _next_agent_order_id(self) -> int:
        """Generate unique order ID for the RL agent."""
        self._agent_order_id_counter += 1
        return self._agent_order_id_counter

    def _next_opp_order_id(self) -> int:
        """Generate unique order ID for the opponent."""
        self._opp_order_id_counter += 1
        return self._opp_order_id_counter

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset environment: new engine, sample opponent, seed book.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)

        if seed is not None:
            self._rng = random.Random(seed)

        # Sample opponent for this episode
        if self._current_opponent is not None:
            opponent = self._current_opponent
            self._current_opponent = None  # Reset for next episode
        else:
            opponent = self._sample_opponent()
        self._active_opponent = opponent

        # Reset engine
        self._engine = ex.MatchingEngine()
        self._step_count = 0

        # Reset agent state
        self._agent_inventory = 0
        self._agent_pnl = 0.0
        self._prev_agent_pnl = 0.0
        self._agent_order_id_counter = 900_000_000
        self._my_order_ids = set()

        # Reset opponent state
        self._opp_inventory = 0
        self._opp_pnl = 0.0
        self._prev_opp_pnl = 0.0
        self._opp_order_id_counter = 800_000_000
        self._opp_order_ids = set()

        # Reset fill routing
        self._order_to_agent = {}

        # Create noise traders
        self._noise_traders = []
        for i in range(self.num_noise_traders):
            trader_seed = self._rng.randint(0, 2**31)
            trader = RandomAgent(agent_id=i + 1, tick_range=5, seed=trader_seed)
            self._noise_traders.append(trader)

        # Warmup: seed the book
        for warmup_step in range(50):
            timestamp = warmup_step * 1_000_000
            self._run_noise_traders(timestamp)

        # Reward baselines, set after warmup. Both zero, nothing is held yet.
        self._prev_agent_equity = self._marked_equity(0, 0.0)
        self._prev_opp_equity = self._marked_equity(0, 0.0)

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step: RL agent acts, opponent acts, noise traders act.

        Args:
            action: RL agent's action.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        assert self._engine is not None, "Must call reset() before step()"

        self._step_count += 1
        timestamp = (50 + self._step_count) * 1_000_000

        # 1. Execute RL agent's action
        self._execute_action(action, timestamp, is_opponent=False)

        # 2. Execute opponent's action
        opp_obs = self._get_observation_for_opponent()
        opp_action = self._active_opponent.predict(opp_obs)
        self._execute_action(opp_action, timestamp, is_opponent=True)

        # 3. Run noise traders
        self._run_noise_traders(timestamp)

        # 4. Compute reward (competitive), on marked equity rather than cash.
        # See _marked_equity: the cash delta rewards selling by the full notional
        # and punishes buying the same way, which is not profit.
        agent_equity = self._marked_equity(self._agent_inventory, self._agent_pnl)
        opp_equity = self._marked_equity(self._opp_inventory, self._opp_pnl)
        agent_pnl_delta = agent_equity - self._prev_agent_equity
        opp_pnl_delta = opp_equity - self._prev_opp_equity
        self._prev_agent_equity = agent_equity
        self._prev_opp_equity = opp_equity
        inv_penalty = self.inventory_penalty * abs(self._agent_inventory)

        # Blend absolute and relative reward
        absolute_reward = agent_pnl_delta - inv_penalty
        relative_reward = (agent_pnl_delta - opp_pnl_delta) - inv_penalty
        reward = float(
            (1 - self.competitive_reward_weight) * absolute_reward
            + self.competitive_reward_weight * relative_reward
        )

        self._prev_agent_pnl = self._agent_pnl
        self._prev_opp_pnl = self._opp_pnl

        # Check termination
        terminated = False
        truncated = self._step_count >= self.episode_length

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _mid_price(self) -> float:
        """Current mid in fixed-point, falling back to the last usable price."""
        book = self._engine.book()
        best_bid = book.best_bid_price()
        best_ask = book.best_ask_price()
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        if best_bid is not None:
            return float(best_bid)
        if best_ask is not None:
            return float(best_ask)
        return float(self._BASE_PRICE)

    def _marked_equity(self, inventory: int, pnl: float) -> float:
        """Cash plus inventory valued at the mid.

        Both the reward and the win/loss call need this. Comparing raw cash
        across two agents decides the match on who is holding less stock rather
        than on who traded better: the one left long has paid out cash for
        something the comparison does not count.
        """
        return pnl + inventory * self._mid_price()

    def _execute_action(
        self, action: int, timestamp: int, is_opponent: bool
    ) -> None:
        """Execute an action for either the RL agent or opponent.

        Args:
            action: Discrete action (0-4).
            timestamp: Current timestamp.
            is_opponent: If True, execute for opponent.
        """
        if action == 0:
            return

        book = self._engine.book()
        best_bid = book.best_bid_price()
        best_ask = book.best_ask_price()

        if best_bid is None or best_ask is None:
            return

        order = ex.Order()
        if is_opponent:
            order.id = self._next_opp_order_id()
        else:
            order.id = self._next_agent_order_id()
        order.quantity = self.order_quantity
        order.timestamp = timestamp
        order.filled_quantity = 0
        order.stop_price = 0
        order.peg_offset = 0
        order.visible_quantity = 0
        order.hidden_quantity = 0

        if action == 1:  # buy limit at bid
            order.side = ex.Side.Buy
            order.price = best_bid
            order.type = ex.OrderType.Limit
            order.tif = ex.TimeInForce.GTC
        elif action == 2:  # buy market
            order.side = ex.Side.Buy
            order.price = best_ask
            order.type = ex.OrderType.Limit
            order.tif = ex.TimeInForce.IOC
        elif action == 3:  # sell limit at ask
            order.side = ex.Side.Sell
            order.price = best_ask
            order.type = ex.OrderType.Limit
            order.tif = ex.TimeInForce.GTC
        elif action == 4:  # sell market
            order.side = ex.Side.Sell
            order.price = best_bid
            order.type = ex.OrderType.Limit
            order.tif = ex.TimeInForce.IOC
        else:
            return

        if is_opponent:
            self._opp_order_ids.add(order.id)
        else:
            self._my_order_ids.add(order.id)

        fills = self._engine.submit(order)
        self._process_fills(fills)

    def _process_fills(self, fills: list) -> None:
        """Route fills to the correct agent (RL agent or opponent)."""
        for fill in fills:
            taker_is_agent = fill.taker_order_id in self._my_order_ids
            maker_is_agent = fill.maker_order_id in self._my_order_ids
            taker_is_opp = fill.taker_order_id in self._opp_order_ids
            maker_is_opp = fill.maker_order_id in self._opp_order_ids

            # Update RL agent
            if taker_is_agent:
                if fill.aggressor_side.name == "Buy":
                    self._agent_inventory += fill.quantity
                    self._agent_pnl -= fill.price * fill.quantity
                else:
                    self._agent_inventory -= fill.quantity
                    self._agent_pnl += fill.price * fill.quantity
            if maker_is_agent:
                if fill.aggressor_side.name == "Buy":
                    self._agent_inventory -= fill.quantity
                    self._agent_pnl += fill.price * fill.quantity
                else:
                    self._agent_inventory += fill.quantity
                    self._agent_pnl -= fill.price * fill.quantity

            # Update opponent
            if taker_is_opp:
                if fill.aggressor_side.name == "Buy":
                    self._opp_inventory += fill.quantity
                    self._opp_pnl -= fill.price * fill.quantity
                else:
                    self._opp_inventory -= fill.quantity
                    self._opp_pnl += fill.price * fill.quantity
            if maker_is_opp:
                if fill.aggressor_side.name == "Buy":
                    self._opp_inventory -= fill.quantity
                    self._opp_pnl += fill.price * fill.quantity
                else:
                    self._opp_inventory += fill.quantity
                    self._opp_pnl -= fill.price * fill.quantity

            # Route to noise traders
            taker_agent = self._order_to_agent.get(fill.taker_order_id)
            if taker_agent is not None:
                taker_agent.on_fill(fill)
            maker_agent = self._order_to_agent.get(fill.maker_order_id)
            if maker_agent is not None:
                maker_agent.on_fill(fill)

    def _run_noise_traders(self, timestamp: int) -> None:
        """Run noise traders and process fills."""
        for trader in self._noise_traders:
            orders = trader.on_market_data(self._engine, timestamp)
            for order in orders:
                self._order_to_agent[order.id] = trader
                fills = self._engine.submit(order)
                self._process_fills(fills)

    def _get_observation(self) -> np.ndarray:
        """Get observation for the RL agent."""
        return self._build_observation(
            self._agent_inventory, self._agent_pnl
        )

    def _get_observation_for_opponent(self) -> np.ndarray:
        """Get observation for the opponent (sees same book, own inventory)."""
        return self._build_observation(
            self._opp_inventory, self._opp_pnl
        )

    def _build_observation(
        self, inventory: int, pnl: float
    ) -> np.ndarray:
        """Construct observation vector for a given agent's state.

        Args:
            inventory: Agent's current inventory.
            pnl: Agent's current PnL.

        Returns:
            numpy array of shape (11,).
        """
        book = self._engine.book()
        best_bid = book.best_bid_price()
        best_ask = book.best_ask_price()

        if best_bid is not None and best_ask is not None:
            mid_price = (best_bid + best_ask) / 2.0
            spread = best_ask - best_bid
        elif best_bid is not None:
            mid_price = float(best_bid)
            spread = 0.0
        elif best_ask is not None:
            mid_price = float(best_ask)
            spread = 0.0
        else:
            mid_price = float(self._BASE_PRICE)
            spread = 0.0

        norm_mid = (mid_price - self._BASE_PRICE) / self._BASE_PRICE
        norm_spread = spread / 1000.0

        total_bid_depth = book.bid_depth()
        total_ask_depth = book.ask_depth()

        bid_levels = self._distribute_depth(total_bid_depth, 3)
        ask_levels = self._distribute_depth(total_ask_depth, 3)

        norm_bid_levels = [d / 50.0 for d in bid_levels]
        norm_ask_levels = [d / 50.0 for d in ask_levels]

        norm_inventory = inventory / 100.0
        norm_pnl = pnl / 1_000_000.0

        total_depth = total_bid_depth + total_ask_depth
        if total_depth > 0:
            imbalance = (total_bid_depth - total_ask_depth) / total_depth
        else:
            imbalance = 0.0

        obs = np.array(
            [
                norm_mid,
                norm_spread,
                norm_bid_levels[0],
                norm_bid_levels[1],
                norm_bid_levels[2],
                norm_ask_levels[0],
                norm_ask_levels[1],
                norm_ask_levels[2],
                norm_inventory,
                norm_pnl,
                imbalance,
            ],
            dtype=np.float32,
        )
        return obs

    @staticmethod
    def _distribute_depth(total: int, levels: int) -> list[float]:
        """Distribute total depth across simulated levels."""
        if total == 0:
            return [0.0] * levels
        weights = list(range(levels, 0, -1))
        weight_sum = sum(weights)
        return [total * w / weight_sum for w in weights]

    def _get_info(self) -> dict:
        """Return info dict with agent and opponent stats."""
        return {
            "inventory": self._agent_inventory,
            "pnl": self._agent_pnl,
            "step": self._step_count,
            "opponent_inventory": self._opp_inventory,
            "opponent_pnl": self._opp_pnl,
            "opponent_name": getattr(self._active_opponent, "name", "unknown"),
        }

    def get_episode_result(self) -> dict:
        """Get end-of-episode results for win/loss tracking.

        PnL is marked to market at the closing mid, so a player holding stock at
        the end is judged on what it is worth rather than on the cash it spent.

        Returns:
            Dict with agent_pnl, opponent_pnl, and win flag.
        """
        agent_equity = self._marked_equity(self._agent_inventory, self._agent_pnl)
        opp_equity = self._marked_equity(self._opp_inventory, self._opp_pnl)
        return {
            "agent_pnl": agent_equity,
            "opponent_pnl": opp_equity,
            "agent_cash": self._agent_pnl,
            "opponent_cash": self._opp_pnl,
            "agent_inventory": self._agent_inventory,
            "opponent_inventory": self._opp_inventory,
            "agent_won": agent_equity > opp_equity,
            "opponent_name": getattr(self._active_opponent, "name", "unknown"),
        }
