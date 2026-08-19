"""Avellaneda-Stoikov market maker."""

import math

from .base import BaseAgent

# Prices are fixed-point with four decimals, so one tick is 0.0001 price units
# and 100.0000 is 100_0000. The AS formulas are in price units and the book is
# in ticks, so every crossing between the two goes through this.
TICKS_PER_UNIT = 10000.0


class MarketMakerAgent(BaseAgent):
    """Avellaneda-Stoikov market maker.

    Quotes are placed at the reservation price plus and minus half the optimal
    spread:

        r     = mid - q*gamma*sigma^2*tau
        delta = gamma*sigma^2*tau + (2/gamma)*ln(1 + gamma/k)

    with an aggressive unwind once inventory passes ``max_inventory``, which the
    model itself does not provide and which exists because the skew goes to zero
    as tau does.

    The defaults are calibrated for this simulator's price scale, which matters
    more than it sounds. The formulas are in price units and one tick is 0.0001.
    Measured over 10 seeds of 3000 steps against three RandomAgents:

        per-step stdev of the mid   0.483 ticks, so 0.0000483 in price units
        mid moves on               21.0% of steps
        mean book spread            2.03 ticks
        quantity at the touch      20.8 units across 34 price levels

    From those:

        sigma  0.000048  the measured per-step stdev of the mid
        gamma  5700      set so the skew reaches ~2 ticks at q = max_inventory
                         while tau is still near 1
        k      8100      set so delta comes out near two ticks, which is the mean
                         book spread, so r +/- delta/2 lands at the touch
        dt     0.0001    1/10000, so tau runs from 1 down to dt over the default
                         10000-step horizon rather than hitting the floor early

    These replace two earlier sets, both fitted to a market that did not work.
    RandomAgent never cancelled its orders and rested them 100x too far out, so
    the mid took 13 distinct values in 3000 steps and this agent got 3 fills.
    Measured volatility there was 0.000234, which was 13 jumps averaged over 2987
    frozen steps. Fixing only the cancellation gave a book whose per-step
    volatility was 3x its own spread, where no quoter can profit. See the
    RandomAgent docstring for the numbers.

    Against a naive quoter that just rests at the touch on both sides and dumps
    inventory past a limit, over 10 seeds of 3000 steps:

        agent                  equity        fills  passive  mean|inv|  max|inv|
        Avellaneda-Stoikov  394.6 +/- 161     222     89%       3.3       13.9
        touch quoter        388.7 +/- 152     241     91%       8.2       18.8

    So the model does not earn more. It earns the same and carries 2.5x less
    inventory, winning on equity in 4 of 10 seeds. That is the honest reading and
    it is what the model is for: the skew is a risk control, not an alpha source,
    and against uninformed flow there is no adverse selection for it to dodge.
    Expect the gap to open up if the noise traders are ever given a direction.

    Getting these wrong does not fail loudly, it just stops the agent trading.
    An earlier version of this class shipped with gamma=0.1, sigma=0.002, k=1.5,
    which are reasonable for a stock quoted in dollars and give a half-spread of
    6454 ticks here, so no quote could ever have filled, and a skew of 0.0003
    ticks, which rounds to nothing at all. ``TestQuoteScale`` in
    ``test_market_maker.py`` pins the half-spread and the skew to a tick range so
    a future change of price scale fails a test instead of going quiet.

    Parameters:
        gamma: Risk aversion coefficient (higher = tighter inventory control).
        sigma: Per-step volatility of the mid, in price units.
        k: Order arrival intensity decay, in inverse price units.
        dt: Time step as a fraction of the total trading horizon.
        quantity: Order size per side.
        max_inventory: Inventory threshold for aggressive unwinding.
        edge_ticks: Minimum profit (in ticks) required to unwind inventory.
    """

    def __init__(
        self,
        agent_id: int,
        gamma: float = 5700.0,
        sigma: float = 0.000048,
        k: float = 8100.0,
        dt: float = 0.0001,
        quantity: int = 5,
        max_inventory: int = 15,
        edge_ticks: int = 2,
    ):
        super().__init__(agent_id)
        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.dt = dt
        self.quantity = quantity
        self.max_inventory = max_inventory
        self.edge_ticks = edge_ticks

        # Track step count
        self._step = 0

        # Track outstanding order IDs so we can cancel them
        self._outstanding_order_ids: list[int] = []

        # Midprice history
        self._mid_history: list[float] = []

        # Track cost basis for inventory management
        self._total_buy_cost: int = 0  # in fixed-point * quantity
        self._total_buy_qty: int = 0
        self._total_sell_revenue: int = 0
        self._total_sell_qty: int = 0

    @property
    def name(self) -> str:
        return f"MM-{self.agent_id}"

    def on_fill(self, fill) -> None:
        """Track fills for inventory cost basis."""
        super().on_fill(fill)

        is_maker = fill.maker_order_id in self._my_order_ids
        is_taker = fill.taker_order_id in self._my_order_ids

        if is_maker:
            if fill.aggressor_side.name == "Sell":
                # We're the maker on the buy side (someone sold to us)
                self._total_buy_cost += fill.price * fill.quantity
                self._total_buy_qty += fill.quantity
            else:
                # We're the maker on the sell side (someone bought from us)
                self._total_sell_revenue += fill.price * fill.quantity
                self._total_sell_qty += fill.quantity

        if is_taker:
            if fill.aggressor_side.name == "Buy":
                # We aggressively bought
                self._total_buy_cost += fill.price * fill.quantity
                self._total_buy_qty += fill.quantity
            else:
                # We aggressively sold
                self._total_sell_revenue += fill.price * fill.quantity
                self._total_sell_qty += fill.quantity

    def _avg_buy_price(self) -> int:
        """Average purchase price in fixed-point."""
        if self._total_buy_qty == 0:
            return 0
        return self._total_buy_cost // self._total_buy_qty

    def on_market_data(self, engine, timestamp: int) -> list:
        """Compute quotes and submit orders.

        Each step:
          1. Cancel previous outstanding orders.
          2. Read current book state.
          3. Compute the reservation price and the optimal spread.
          4. Place bid at r - delta/2.
          5. Place ask at r + delta/2, and/or aggressive unwind.
        """
        import exchange_simulator as ex

        # --- Step 1: Cancel previous orders ---
        for oid in self._outstanding_order_ids:
            engine.cancel(oid)
        self._outstanding_order_ids = []

        self._step += 1

        # --- Step 2: Read book state ---
        book = engine.book()
        best_bid = book.best_bid_price()
        best_ask = book.best_ask_price()

        orders = []

        if best_bid is None or best_ask is None:
            # No two-sided book yet; seed with a reasonable spread
            base_price = 100_0000  # 100.0000 in fixed-point
            if best_bid is None:
                o = self.make_order(
                    ex.Side.Buy, base_price - 500, self.quantity,
                    ex.OrderType.Limit, ex.TimeInForce.GTC, timestamp
                )
                orders.append(o)
                self._outstanding_order_ids.append(o.id)
            if best_ask is None:
                o = self.make_order(
                    ex.Side.Sell, base_price + 500, self.quantity,
                    ex.OrderType.Limit, ex.TimeInForce.GTC, timestamp
                )
                orders.append(o)
                self._outstanding_order_ids.append(o.id)
            return orders

        # Midprice in price units. Not integer-divided: the book spread here is
        # one tick almost all of the time, so (bb + ba) // 2 would round the mid
        # down onto the best bid and cost half a tick. At a half-tick
        # half-spread that is the difference between quoting at the touch and
        # quoting behind it.
        mid = (best_bid + best_ask) / 2.0 / TICKS_PER_UNIT

        self._mid_history.append(mid)

        # --- Step 3: Avellaneda-Stoikov reservation price and optimal spread ---
        q = self.inventory
        tau = max(self.dt, 1.0 - self._step * self.dt)

        # Reservation price: the mid shifted against inventory. Long inventory
        # pushes it down, so the bid retreats and the ask comes in.
        r = mid - q * self.gamma * (self.sigma ** 2) * tau

        # Optimal spread. The second term dominates and is what sets the width;
        # see the class docstring for how gamma, sigma and k were calibrated to
        # make it land near one tick on this price scale.
        delta = (
            self.gamma * (self.sigma ** 2) * tau
            + (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)
        )
        half = 0.5 * delta

        # --- Step 4: Place bid at the AS quote ---
        # Round the bid down and the ask up, rather than to nearest. Two
        # reasons. It is the conservative direction for a quote on each side,
        # and it guarantees the two land on different ticks: delta is a little
        # under one tick here, so rounding both to nearest can put the bid and
        # the ask on the same price, which is not a quote.
        bid_price = math.floor((r - half) * TICKS_PER_UNIT)
        bid_price = max(bid_price, 100)

        # No clamp back inside the touch. When inventory is large enough that
        # the skew exceeds a tick, the model is asking to cross, and a limit
        # order through the touch just fills at the touch. That is the model
        # unwinding, and neutralizing it here would compute a skew and then
        # discard it, which is what this agent used to do with the whole
        # formula. Note that a sub-tick skew cannot move an integer price at
        # all, so the skew only starts to bite around q = 8.

        # Only place bid if inventory isn't too large
        if q < self.max_inventory:
            bid_order = self.make_order(
                ex.Side.Buy, bid_price, self.quantity,
                ex.OrderType.Limit, ex.TimeInForce.GTC, timestamp
            )
            orders.append(bid_order)
            self._outstanding_order_ids.append(bid_order.id)

        # --- Step 5: Place ask at the AS quote, plus aggressive unwind ---
        ask_price = math.ceil((r + half) * TICKS_PER_UNIT)

        ask_order = self.make_order(
            ex.Side.Sell, ask_price, self.quantity,
            ex.OrderType.Limit, ex.TimeInForce.GTC, timestamp
        )
        orders.append(ask_order)
        self._outstanding_order_ids.append(ask_order.id)

        # Aggressive inventory unwind when we're too long
        # Sell at best_bid only if profitable (above avg purchase price + edge)
        if q > self.max_inventory:
            avg_buy = self._avg_buy_price()
            target_sell = avg_buy + self.edge_ticks if avg_buy > 0 else best_bid

            # Sell aggressively if we can still make money, or if inventory is
            # dangerously high (accept a small loss to reduce risk)
            sell_price = best_bid
            if q > self.max_inventory * 2:
                # Emergency unwind: just sell at bid regardless
                unwind_qty = min(self.quantity, q - self.max_inventory)
                sell_order = self.make_order(
                    ex.Side.Sell, sell_price, unwind_qty,
                    ex.OrderType.Limit, ex.TimeInForce.GTC, timestamp
                )
                orders.append(sell_order)
                self._outstanding_order_ids.append(sell_order.id)
            elif best_bid >= target_sell:
                # Profitable unwind
                unwind_qty = min(self.quantity, q - self.max_inventory // 2)
                sell_order = self.make_order(
                    ex.Side.Sell, sell_price, unwind_qty,
                    ex.OrderType.Limit, ex.TimeInForce.GTC, timestamp
                )
                orders.append(sell_order)
                self._outstanding_order_ids.append(sell_order.id)

        # Symmetric: aggressive buy unwind when too short
        elif q < -self.max_inventory:
            buy_price = best_ask
            if q < -self.max_inventory * 2:
                unwind_qty = min(self.quantity, -q - self.max_inventory)
                buy_order = self.make_order(
                    ex.Side.Buy, buy_price, unwind_qty,
                    ex.OrderType.Limit, ex.TimeInForce.GTC, timestamp
                )
                orders.append(buy_order)
                self._outstanding_order_ids.append(buy_order.id)

        return orders
