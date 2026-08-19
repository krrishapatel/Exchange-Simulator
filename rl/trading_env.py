"""Gymnasium environment wrapping the exchange simulator matching engine."""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import exchange_simulator as ex
from agents.random_agent import RandomAgent


class TradingEnv(gym.Env):
    """A Gymnasium environment for training RL trading agents.

    The agent interacts with a C++ matching engine populated by background
    noise traders (RandomAgents) that provide liquidity.

    Observation space (Box, float32, shape=(11,)):
        [0] mid, as (mid - 100.0000) in ticks / 10
        [1] book spread in ticks / 5
        [2-4] bid quantity resting at levels 0, 1, 2 / 20
        [5-7] ask quantity resting at levels 0, 1, 2 / 20
        [8] inventory / max_inventory, so [-1, 1] when a limit is set
        [9] marked equity in ticks / 100
        [10] touch imbalance, (bid_qty_0 - ask_qty_0) / (bid_qty_0 + ask_qty_0)

    Divisors come from measuring the default market over 5 seeds of 1000 steps,
    so a typical value of each feature is order 1: mid range 12.3 ticks per
    episode, mean spread 1.81, per-level bid quantity 11.2 / 15.1 / 16.2.

    Features [2-7] used to be fake. ``OrderBook`` had no per-level accessor, so
    the env called ``bid_depth()``, which counts price levels and not quantity,
    then split that single number across three "levels" with fixed 3:2:1 weights.
    Levels 1 and 2 carried no information the sum did not already carry, the
    units were wrong, and [10] was a level-count imbalance rather than a
    quantity imbalance. ``l2_bids``/``l2_asks`` were added to the engine so these
    features can be what they claim to be.

    Action space (Discrete(5)):
        0: hold
        1: buy_limit_at_bid
        2: buy_market
        3: sell_limit_at_ask
        4: sell_market

    Actions 1 and 3 rest for one step and are then cancelled, the same requoting
    market_maker.py does. They used to be GTC and never cancelled, so an agent
    that quoted every step left every unfilled quote in the book: over 1000 steps
    of action 1 the bid side went from 375 units in 67 orders to 746 in 438, half
    of it the agent's own stale quotes, and the agent was then marked against a
    book it had built.

    Reward:
        Change in marked equity, minus ``inventory_penalty`` * abs(inventory).
    """

    metadata = {"render_modes": []}

    # Fixed-point price constants (1 unit = 0.0001 in real terms)
    _BASE_PRICE = 100_0000  # 100.0000

    # Observation normalizers, measured on the default market. See the class
    # docstring for where each comes from.
    _MID_SCALE = 10.0
    _SPREAD_SCALE = 5.0
    _LEVEL_QTY_SCALE = 20.0
    _EQUITY_SCALE = 100.0

    def __init__(
        self,
        num_noise_traders: int = 5,
        episode_length: int = 1000,
        inventory_penalty: float = 0.005,
        order_quantity: int = 1,
        max_inventory: int = 25,
        seed: int | None = None,
    ):
        """Initialize the trading environment.

        Args:
            num_noise_traders: Number of background RandomAgent noise traders.
            episode_length: Number of steps per episode.
            inventory_penalty: Per-step penalty on abs(inventory), in ticks.
                The mid is a random walk here, so holding inventory has zero
                expected payoff and any penalty makes an idle position bad. Its
                real job is to make sitting at the limit hurt: at the default,
                q = max_inventory costs 125 ticks over a 1000-step episode,
                against a market-making edge of roughly 25 ticks at size 1.
                A small position (q = 3) costs 15. The old default of 0.001 was
                0.3% of the per-step mid noise at q = 10, so it did not bind.
            order_quantity: Quantity for RL agent orders.
            max_inventory: Position limit. Orders that would breach it are
                dropped, so the agent holds at most this many units long or
                short. Set to 0 for no limit, which is only useful for showing
                what happens without one: always-buy-market reached 629 units on
                a book whose touch holds about 11, and was then marked at a mid
                it was itself holding up. The limit is exact rather than
                approximate because the agent has at most one live order.
            seed: Optional random seed for reproducibility.
        """
        super().__init__()

        self.num_noise_traders = num_noise_traders
        self.episode_length = episode_length
        self.inventory_penalty = inventory_penalty
        self.order_quantity = order_quantity
        self.max_inventory = max_inventory
        self._seed = seed

        # Observation: 11 floats
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )

        # Actions: hold, buy_limit_at_bid, buy_market, sell_limit_at_ask, sell_market
        self.action_space = spaces.Discrete(5)

        # Internal state (set during reset)
        self._engine: ex.MatchingEngine | None = None
        self._noise_traders: list[RandomAgent] = []
        self._step_count = 0
        self._agent_inventory = 0
        self._agent_pnl = 0.0
        self._prev_pnl = 0.0
        self._prev_equity = 0.0
        self._order_id_counter = 0
        self._my_order_ids: set[int] = set()
        self._live_order_ids: list[int] = []
        self._order_to_agent: dict[int, RandomAgent] = {}
        self._rng = random.Random(seed)

    def _next_order_id(self) -> int:
        """Generate the next unique order ID for the RL agent."""
        self._order_id_counter += 1
        return self._order_id_counter

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to a fresh state.

        Creates a new matching engine, initializes noise traders, and seeds
        the order book with initial orders.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)

        if seed is not None:
            self._rng = random.Random(seed)

        self._engine = ex.MatchingEngine()
        self._step_count = 0
        self._agent_inventory = 0
        self._agent_pnl = 0.0
        self._prev_pnl = 0.0
        self._prev_equity = 0.0
        self._order_id_counter = 900_000_000  # High range to avoid collisions
        self._my_order_ids = set()
        self._live_order_ids = []
        self._order_to_agent = {}

        # Create noise traders with unique seeds
        self._noise_traders = []
        for i in range(self.num_noise_traders):
            trader_seed = self._rng.randint(0, 2**31)
            trader = RandomAgent(agent_id=i + 1, seed=trader_seed)
            self._noise_traders.append(trader)

        # Seed the book: run noise traders for a warmup period
        for warmup_step in range(50):
            timestamp = warmup_step * 1_000_000
            self._run_noise_traders(timestamp)

        # Baseline for the reward, set after warmup so the first step is not
        # charged for the book appearing. Zero here, since nothing is held yet.
        self._prev_equity = self._marked_equity()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment.

        Args:
            action: Integer action from the action space.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        assert self._engine is not None, "Must call reset() before step()"

        self._step_count += 1
        timestamp = (50 + self._step_count) * 1_000_000  # Continue after warmup

        # Execute the RL agent's action
        self._execute_action(action, timestamp)

        # Run noise traders for this step
        self._run_noise_traders(timestamp)

        # Reward is the change in marked equity, not the change in cash.
        equity = self._marked_equity()
        inv_penalty = self.inventory_penalty * abs(self._agent_inventory)
        reward = float(equity - self._prev_equity - inv_penalty)
        self._prev_equity = equity
        self._prev_pnl = self._agent_pnl

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

    def _marked_equity(self) -> float:
        """Cash plus inventory valued at the mid.

        ``_agent_pnl`` is cash flow only: a buy subtracts the full notional and a
        sell adds it, so it says nothing about whether a trade was good. Valuing
        the position closes that gap, and the difference is not subtle. Rewarding
        the cash delta directly ranked the fixed policies close to backwards over
        500 steps: always-buy-market scored -500,005,283 while actually being the
        most profitable policy at +31,592 marked, and always-sell-market scored
        +499,927,312 for half that profit. An agent trained on the cash delta
        learns to sell whatever it holds and never buy, because selling pays the
        full notional up front and the reward never accounts for what was given
        away.
        """
        return self._agent_pnl + self._agent_inventory * self._mid_price()

    def _execute_action(self, action: int, timestamp: int) -> None:
        """Translate the discrete action into an order and submit it.

        Cancels last step's quote first, so a resting order lives exactly one
        step. Orders that would breach ``max_inventory`` are dropped, which the
        agent sees as a hold.

        Actions:
            0: hold (do nothing)
            1: buy limit at best bid
            2: buy market (cross the spread)
            3: sell limit at best ask
            4: sell market (cross the spread)
        """
        self._cancel_live_orders()

        if action == 0:
            return

        # Position limit. Buys are 1 and 2, sells are 3 and 4.
        if self.max_inventory:
            if action in (1, 2) and self._agent_inventory >= self.max_inventory:
                return
            if action in (3, 4) and self._agent_inventory <= -self.max_inventory:
                return

        book = self._engine.book()
        best_bid = book.best_bid_price()
        best_ask = book.best_ask_price()

        # Need a valid book to place orders
        if best_bid is None or best_ask is None:
            return

        order = ex.Order()
        order.id = self._next_order_id()
        order.quantity = self.order_quantity
        order.timestamp = timestamp
        order.filled_quantity = 0
        order.stop_price = 0
        order.peg_offset = 0
        order.visible_quantity = 0
        order.hidden_quantity = 0

        if action == 1:
            # Buy limit at bid
            order.side = ex.Side.Buy
            order.price = best_bid
            order.type = ex.OrderType.Limit
            order.tif = ex.TimeInForce.GTC
        elif action == 2:
            # Buy market (use IOC at ask price to cross)
            order.side = ex.Side.Buy
            order.price = best_ask
            order.type = ex.OrderType.Limit
            order.tif = ex.TimeInForce.IOC
        elif action == 3:
            # Sell limit at ask
            order.side = ex.Side.Sell
            order.price = best_ask
            order.type = ex.OrderType.Limit
            order.tif = ex.TimeInForce.GTC
        elif action == 4:
            # Sell market (use IOC at bid price to cross)
            order.side = ex.Side.Sell
            order.price = best_bid
            order.type = ex.OrderType.Limit
            order.tif = ex.TimeInForce.IOC
        else:
            return

        self._my_order_ids.add(order.id)
        if order.tif == ex.TimeInForce.GTC:
            self._live_order_ids.append(order.id)
        fills = self._engine.submit(order)
        self._process_rl_agent_fills(fills)

    def _cancel_live_orders(self) -> None:
        """Pull last step's resting quote out of the book.

        Cancelling something that already filled is not an error, the engine
        reports AlreadyFilled, so filled orders need no separate bookkeeping.
        """
        for order_id in self._live_order_ids:
            self._engine.cancel(order_id)
        self._live_order_ids = []

    def _process_rl_agent_fills(self, fills: list) -> None:
        """Update RL agent inventory and PnL based on fills."""
        for fill in fills:
            is_taker = fill.taker_order_id in self._my_order_ids
            is_maker = fill.maker_order_id in self._my_order_ids

            if is_taker:
                if fill.aggressor_side.name == "Buy":
                    self._agent_inventory += fill.quantity
                    self._agent_pnl -= fill.price * fill.quantity
                else:
                    self._agent_inventory -= fill.quantity
                    self._agent_pnl += fill.price * fill.quantity

            if is_maker:
                if fill.aggressor_side.name == "Buy":
                    # Aggressor bought from us, we sold
                    self._agent_inventory -= fill.quantity
                    self._agent_pnl += fill.price * fill.quantity
                else:
                    # Aggressor sold to us, we bought
                    self._agent_inventory += fill.quantity
                    self._agent_pnl -= fill.price * fill.quantity

    def _run_noise_traders(self, timestamp: int) -> None:
        """Run all noise traders for one step and process their fills."""
        for trader in self._noise_traders:
            orders = trader.on_market_data(self._engine, timestamp)
            for order in orders:
                self._order_to_agent[order.id] = trader
                fills = self._engine.submit(order)
                for fill in fills:
                    # Route fills to noise traders
                    taker_agent = self._order_to_agent.get(fill.taker_order_id)
                    if taker_agent is not None:
                        taker_agent.on_fill(fill)
                    maker_agent = self._order_to_agent.get(fill.maker_order_id)
                    if maker_agent is not None:
                        maker_agent.on_fill(fill)
                    # Also check if RL agent is involved
                    self._process_rl_agent_fills([fill])

    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector from current market state.

        Returns:
            numpy array of shape (11,) with float32 values.
        """
        book = self._engine.book()
        bid_levels = book.l2_bids(3)
        ask_levels = book.l2_asks(3)

        if bid_levels and ask_levels:
            mid_price = (bid_levels[0].price + ask_levels[0].price) / 2.0
            spread = float(ask_levels[0].price - bid_levels[0].price)
        elif bid_levels:
            mid_price = float(bid_levels[0].price)
            spread = 0.0
        elif ask_levels:
            mid_price = float(ask_levels[0].price)
            spread = 0.0
        else:
            mid_price = float(self._BASE_PRICE)
            spread = 0.0

        # Quantity resting at each of the top three levels. A side thinner than
        # three levels reports 0 for the missing ones, which is what an empty
        # level holds.
        bid_qty = [0.0, 0.0, 0.0]
        ask_qty = [0.0, 0.0, 0.0]
        for i, level in enumerate(bid_levels):
            bid_qty[i] = float(level.total_quantity)
        for i, level in enumerate(ask_levels):
            ask_qty[i] = float(level.total_quantity)

        norm_mid = (mid_price - self._BASE_PRICE) / self._MID_SCALE
        norm_spread = spread / self._SPREAD_SCALE
        norm_bid_qty = [q / self._LEVEL_QTY_SCALE for q in bid_qty]
        norm_ask_qty = [q / self._LEVEL_QTY_SCALE for q in ask_qty]

        # Inventory against its own limit, so this is [-1, 1] whenever one is
        # set. Without a limit there is nothing to scale by, so fall back to a
        # constant and let the feature run wherever the position runs.
        inv_scale = float(self.max_inventory) if self.max_inventory else 100.0
        norm_inventory = self._agent_inventory / inv_scale

        # Equity, not cash. obs[9] used to be _agent_pnl, which is cash flow, so
        # it read as a large negative number for any agent holding a position and
        # said nothing about whether that position was good.
        norm_equity = self._marked_equity() / self._EQUITY_SCALE

        # Imbalance from quantity at the touch. This is the feature the roadmap
        # asked for; before l2_bids existed it was a ratio of price-level counts,
        # which moves with how spread out the book is rather than with pressure.
        touch_total = bid_qty[0] + ask_qty[0]
        if touch_total > 0:
            imbalance = (bid_qty[0] - ask_qty[0]) / touch_total
        else:
            imbalance = 0.0

        obs = np.array(
            [
                norm_mid,
                norm_spread,
                norm_bid_qty[0],
                norm_bid_qty[1],
                norm_bid_qty[2],
                norm_ask_qty[0],
                norm_ask_qty[1],
                norm_ask_qty[2],
                norm_inventory,
                norm_equity,
                imbalance,
            ],
            dtype=np.float32,
        )
        return obs

    def _get_info(self) -> dict:
        """Return auxiliary info dictionary.

        ``pnl`` is cash flow, ``equity`` is cash plus inventory at the mid. Judge
        a policy on ``equity``; ``pnl`` on its own makes buying look like a loss.
        """
        return {
            "inventory": self._agent_inventory,
            "pnl": self._agent_pnl,
            "equity": self._marked_equity(),
            "step": self._step_count,
        }
