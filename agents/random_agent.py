"""Random agent that places limit orders near the midprice."""

import collections
import math
import random

from .base import BaseAgent


class RandomAgent(BaseAgent):
    """Agent that randomly submits buy/sell limit orders near the midprice.

    Places orders within a configurable number of ticks from the
    current midprice, with random quantities between 1 and 10. A fraction of
    orders cross the spread instead, so the agent trades rather than only
    resting. Resting orders are cancelled after ``order_lifetime`` steps.

    This agent is the market. Everything else in the repo is measured against
    the book it produces, so the three parameters below decide whether any of
    those measurements mean anything. All three were wrong at once, and two of
    them hid each other.

    **Aggression.** Passive orders alone never trade, so crossing has to be
    explicit. An earlier version had no such parameter and got all of its trades
    from a rounding accident: it took the mid as ``(best_bid + best_ask) // 2``,
    which floors onto the bid, so a sell placed at the mid landed exactly on the
    bid and crossed, while a buy placed at the mid sat below the ask could never
    reach it. Sides were still drawn 50/50, so the flow looked balanced while
    every single trade was a sell, and the price walked monotonically down.

    **Order lifetime.** Without cancellation this market has no price discovery.
    Three of these agents add about 16 units of resting size per step and only
    ~8% of flow crosses, so the book thickens monotonically until the touch is a
    wall nobody can clear. Over 3000 steps the touch grew to 3258 units and the
    mid took 13 distinct values, moving on 0.4% of steps. Its per-step stdev
    looked like a healthy 2.28 units, but that was 13 jumps averaged over 2987
    frozen steps.

    **Offset grid.** ``tick_range`` is in ticks and a tick is one fixed-point
    unit, the same convention as market_maker.py. It used to be multiplied by
    100, so ``tick_range=5`` meant resting up to 500 ticks out on a book one tick
    wide. Levels were 100 units apart, so clearing one jumped the mid ~50 units,
    which put per-step volatility at 6.6 ticks against a 2 tick spread. No
    market maker can survive that, and none did.

    Measured over 10 seeds of 3000 steps with a MarketMakerAgent and three of
    these, changing one thing at a time from the original:

        grid  range  lifetime   sigma   moves   spread  levels   MM equity  fills
         100      5     never    2.28    0.4%     1.20    ~119          --      3
         100      5        25    6.67   14.6%     2.16    21.7       -7562    164
          10      5        25    0.74    9.6%     1.29    18.6        -303    100
           1      5        25    0.25    2.6%     1.08    12.1        +152      2
           1     20        25    0.48   21.0%     2.03    33.8        +344    222

    The last row is the default. What makes it a market rather than the rows
    above it: depth is stationary, per-step volatility (0.48 ticks) is below the
    spread (2 ticks) as it is in a real book rather than 3x above it, the mid
    moves often enough to be worth predicting, and a passive market maker earns
    money against uninformed flow instead of being run over.

    Read the equity column loosely. Its standard error across 10 seeds is around
    160, so the last two rows are not separated by it and the first three are only
    separated because they are catastrophic. The columns carrying the argument are
    sigma, the move rate, the spread and the level count.
    """

    def __init__(
        self,
        agent_id: int,
        tick_range: int = 20,
        seed: int | None = None,
        aggression: float = 0.08,
        order_lifetime: int = 25,
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
            order_lifetime: Steps a resting order may live before this agent
                cancels it. Set to 0 to never cancel, which reproduces the
                frozen market described in the class docstring and is only
                useful for demonstrating it.
        """
        super().__init__(agent_id)
        self._tick_range = tick_range
        self._rng = random.Random(seed)
        self._aggression = aggression
        self._order_lifetime = order_lifetime
        # (expiry_step, order_id), oldest first. Lifetime is constant, so
        # appending keeps it sorted and expiry is a pop from the front.
        self._live_orders: collections.deque[tuple[int, int]] = collections.deque()
        self._step = 0

    def _expire_orders(self, engine) -> None:
        """Cancel this agent's orders that have outlived order_lifetime.

        Cancelling an order that already filled is not an error, the engine
        reports AlreadyFilled and nothing changes, so filled orders do not need
        to be tracked separately.
        """
        if not self._order_lifetime:
            return
        while self._live_orders and self._live_orders[0][0] <= self._step:
            engine.cancel(self._live_orders.popleft()[1])

    def _track(self, orders: list) -> list:
        """Record submitted orders so they can be expired later."""
        if self._order_lifetime:
            expiry = self._step + self._order_lifetime
            for order in orders:
                self._live_orders.append((expiry, order.id))
        return orders

    @property
    def name(self) -> str:
        return f"Random-{self.agent_id}"

    def on_market_data(self, engine, timestamp: int) -> list:
        """Generate random orders near the midprice.

        Expires any orders past their lifetime first, then places at most one
        new order. Only places orders when the book has both a bid and an ask.
        Randomly chooses to buy or sell, picks a price within tick_range
        of the midprice, and a quantity between 1 and 10.
        """
        import exchange_simulator as ex

        self._step += 1
        self._expire_orders(engine)

        book = engine.book()

        best_bid = book.best_bid_price()
        best_ask = book.best_ask_price()

        # Only trade when both sides of the book exist
        if best_bid is None or best_ask is None:
            # If book is empty, seed it with a wide spread
            orders = []
            base_price = 100_0000  # 100.0000 in fixed-point
            if best_bid is None:
                bid_price = base_price - self._rng.randint(1, self._tick_range)
                qty = self._rng.randint(1, 10)
                orders.append(
                    self.make_order(
                        ex.Side.Buy, bid_price, qty,
                        ex.OrderType.Limit, ex.TimeInForce.GTC, timestamp
                    )
                )
            if best_ask is None:
                ask_price = base_price + self._rng.randint(1, self._tick_range)
                qty = self._rng.randint(1, 10)
                orders.append(
                    self.make_order(
                        ex.Side.Sell, ask_price, qty,
                        ex.OrderType.Limit, ex.TimeInForce.GTC, timestamp
                    )
                )
            return self._track(orders)

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
            # Rest away from the mid, offset in ticks. One tick is one
            # fixed-point unit, the same convention as market_maker.py.
            offset = self._rng.randint(0, self._tick_range)
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
        return self._track([order])
