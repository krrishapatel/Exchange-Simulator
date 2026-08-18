"""Market maker agent. Named for Avellaneda-Stoikov; see MarketMakerAgent."""

import math

from .base import BaseAgent


class MarketMakerAgent(BaseAgent):
    """Market maker quoting near the touch, with inventory skew and unwinding.

    Quotes are placed at or just inside the best bid and ask, skewed by a
    tick-based function of inventory, with an aggressive unwind once inventory
    passes a threshold. PnL comes from buying passively at the bid and selling
    higher.

    This is NOT the Avellaneda-Stoikov model, despite the class name and the
    parameters. The AS reservation price and optimal spread are computed in
    `on_market_data` and deliberately not used, because at these parameter
    values and this tick size they produce a half-spread of ~6454 ticks and an
    inventory skew below one tick. See the note there. Treat gamma, sigma, k and
    dt as inert until they are recalibrated.

    Parameters:
        gamma: Risk aversion coefficient (higher = tighter inventory control).
        sigma: Estimated volatility of the asset (in price units, float).
        k: Order arrival intensity parameter.
        dt: Time step as a fraction of total trading horizon.
        quantity: Order size per side.
        max_inventory: Inventory threshold for aggressive unwinding.
        edge_ticks: Minimum profit (in ticks) required to unwind inventory.
    """

    def __init__(
        self,
        agent_id: int,
        gamma: float = 0.1,
        sigma: float = 0.002,
        k: float = 1.5,
        dt: float = 0.005,
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
          3. Compute the AS quantities, which are currently unused.
          4. Place bid quote (passive liquidity provision).
          5. Place ask quote (passive) and/or aggressive unwind.
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

        # Midprice in fixed-point and float
        mid_fixed = (best_bid + best_ask) // 2
        mid = mid_fixed / 10000.0
        spread = best_ask - best_bid

        self._mid_history.append(mid)

        # --- Step 3: Avellaneda-Stoikov parameters (not wired in) ---
        q = self.inventory
        tau = max(self.dt, 1.0 - self._step * self.dt)

        # NOTE: the two AS quantities below are computed but not used to place the
        # quotes. They are kept because they are the model this agent is named
        # for, and because the reason they are unused is worth recording rather
        # than deleting.
        #
        # At the defaults (gamma=0.1, sigma=0.002, k=1.5) and this tick size they
        # are not usable as written:
        #
        #   delta = gamma*sigma^2*tau + (2/gamma)*ln(1 + gamma/k)
        #         = 4e-7 + 1.2908, so the second term swamps the first and delta
        #           is ~1.29 price units whatever tau is. Half of that is 6454
        #           ticks, against a book spread of a few ticks, so quotes placed
        #           at r +/- delta/2 would never fill.
        #
        #   r - mid = -q*gamma*sigma^2*tau, which is 0.01 ticks at q=5, so the
        #           inventory skew rounds away to nothing.
        #
        # Making this a real AS agent needs gamma, sigma and k recalibrated for
        # this price scale, not a rewrite of the quoting below. Until then the
        # quoting is the tick-based heuristic in steps 4 and 5, and the class
        # docstring says so.
        r = mid - q * self.gamma * (self.sigma ** 2) * tau  # noqa: F841
        delta = (  # noqa: F841
            self.gamma * (self.sigma ** 2) * tau
            + (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)
        )

        # --- Step 4: Place bid (passive buy) ---
        # Quote at or slightly above best bid to be first in queue
        bid_price = best_bid
        if spread > 2:
            bid_price = best_bid + 1

        # Tick-based inventory skew: when long, lower bid to discourage buying
        if q > 0:
            skew_down = min(q // 5, spread // 3)  # reduce aggressiveness
            bid_price -= skew_down
        elif q < 0:
            # When short, be more aggressive on bid to accumulate
            skew_up = min(-q // 5, spread // 3)
            bid_price += skew_up

        bid_price = max(bid_price, 100)

        # Only place bid if inventory isn't too large
        if q < self.max_inventory:
            bid_order = self.make_order(
                ex.Side.Buy, bid_price, self.quantity,
                ex.OrderType.Limit, ex.TimeInForce.GTC, timestamp
            )
            orders.append(bid_order)
            self._outstanding_order_ids.append(bid_order.id)

        # --- Step 5: Place ask (passive + aggressive unwind) ---
        # Always place a passive ask at or near best_ask
        ask_price = best_ask
        if spread > 2:
            ask_price = best_ask - 1

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
