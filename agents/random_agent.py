"""Random agent that places limit orders near the midprice."""

import math
import random

from .base import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that randomly submits buy/sell limit orders near the midprice.

    Places orders within a configurable number of ticks from the
    current midprice, with random quantities between 1 and 10. A fraction of
    orders cross the spread instead, so the agent trades rather than only
    resting.

    The aggression has to be explicit, because passive orders alone never
    trade. An earlier version had no such parameter and got all of its trades
    from a rounding accident: it took the mid as ``(best_bid + best_ask) // 2``,
    which floors onto the bid, so a sell placed at the mid landed exactly on the
    bid and crossed, while a buy placed at the mid sat below the ask and could
    never reach it. Sides were still drawn 50/50, so the flow looked balanced
    while every single trade was a sell. Measured over 3000 steps: 4557 buys and
    4444 sells submitted, 696 crossings, all of them sells, all at a one-unit
    spread, and the price walked monotonically down. Anything resting on the bid
    got run over and anything resting on the ask never filled.
    """

    def __init__(
        self,
        agent_id: int,
        tick_range: int = 5,
        seed: int | None = None,
        aggression: float = 0.08,
    ):
        """Initialize RandomAgent.

        Args:
            agent_id: Unique agent identifier.
            tick_range: Max distance in ticks from mid for order placement.
            seed: Optional random seed for reproducibility.
            aggression: Probability that an order crosses the spread instead of
                resting. Applied per order after the side is drawn, so buys and
                sells cross equally often. The default matches the 7.7% crossing
                rate the old biased mid produced, so overall trade volume is
                about what it was, just no longer one-directional.
        """
        super().__init__(agent_id)
        self._tick_range = tick_range
        self._rng = random.Random(seed)
        self._aggression = aggression

    @property
    def name(self) -> str:
        return f"Random-{self.agent_id}"

    def on_market_data(self, engine, timestamp: int) -> list:
        """Generate random orders near the midprice.

        Only places orders when the book has both a bid and an ask.
        Randomly chooses to buy or sell, picks a price within tick_range
        of the midprice, and a quantity between 1 and 10.
        """
        import exchange_simulator as ex

        book = engine.book()

        best_bid = book.best_bid_price()
        best_ask = book.best_ask_price()

        # Only trade when both sides of the book exist
        if best_bid is None or best_ask is None:
            # If book is empty, seed it with a wide spread
            orders = []
            base_price = 100_0000  # 100.0000 in fixed-point
            if best_bid is None:
                bid_price = base_price - self._rng.randint(1, self._tick_range) * 100
                qty = self._rng.randint(1, 10)
                orders.append(
                    self.make_order(
                        ex.Side.Buy, bid_price, qty,
                        ex.OrderType.Limit, ex.TimeInForce.GTC, timestamp
                    )
                )
            if best_ask is None:
                ask_price = base_price + self._rng.randint(1, self._tick_range) * 100
                qty = self._rng.randint(1, 10)
                orders.append(
                    self.make_order(
                        ex.Side.Sell, ask_price, qty,
                        ex.OrderType.Limit, ex.TimeInForce.GTC, timestamp
                    )
                )
            return orders

        # Exact mid, deliberately not integer-divided. Flooring here puts the mid
        # on the bid whenever the spread is odd, which is what made this agent a
        # one-way seller. See the class docstring.
        mid = (best_bid + best_ask) / 2.0

        # Randomly choose side
        side = ex.Side.Buy if self._rng.random() < 0.5 else ex.Side.Sell

        if self._rng.random() < self._aggression:
            # Cross the spread. This is the only source of aggression, and it is
            # drawn after the side, so neither side is favoured.
            price = best_ask if side == ex.Side.Buy else best_bid
        else:
            # Rest away from the mid. Price offset in ticks, where 1 tick = 100
            # in fixed-point.
            offset = self._rng.randint(0, self._tick_range) * 100
            # Round away from the mid on each side, so an offset of 0 stays
            # passive instead of landing on the opposite touch.
            if side == ex.Side.Buy:
                price = math.floor(mid - offset)
            else:
                price = math.ceil(mid + offset)

        # Ensure price is positive
        price = max(price, 100)

        quantity = self._rng.randint(1, 10)

        order = self.make_order(
            side, price, quantity,
            ex.OrderType.Limit, ex.TimeInForce.GTC, timestamp
        )
        return [order]
