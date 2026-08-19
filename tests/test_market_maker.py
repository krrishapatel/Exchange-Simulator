"""Tests for the market maker agent.

These cover the Avellaneda-Stoikov quoting, the inventory skew and the unwind.

`TestQuoteScale` is the one to keep if these ever get trimmed. The AS formulas
work in price units and the book is in integer ticks, so a wrong parameter does
not raise, it just puts the quotes thousands of ticks away or makes the skew
round to zero. Those tests pin both to a tick range so that fails loudly.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import exchange_simulator as ex
from agents import MarketMakerAgent, RandomAgent
from simulation import SimulationLoop


def _order(oid, side, price, qty=10):
    o = ex.Order()
    o.id = oid
    o.side = side
    o.price = price
    o.quantity = qty
    o.type = ex.OrderType.Limit
    o.tif = ex.TimeInForce.GTC
    o.timestamp = 0
    o.filled_quantity = 0
    o.stop_price = 0
    o.peg_offset = 0
    o.visible_quantity = 0
    o.hidden_quantity = 0
    return o


def _one_tick_book():
    """A book one tick wide, which is what the simulator produces ~99% of steps."""
    engine = ex.MatchingEngine()
    engine.submit(_order(99001, ex.Side.Buy, 100_0000))
    engine.submit(_order(99002, ex.Side.Sell, 100_0001))
    return engine


class TestMarketMakerBasics:
    """Basic unit tests for MarketMakerAgent."""

    def test_name(self):
        agent = MarketMakerAgent(agent_id=0)
        assert agent.name == "MM-0"

    def test_quotes_both_sides(self):
        """MarketMaker places both a bid and an ask when book is two-sided."""
        engine = ex.MatchingEngine()

        bid = ex.Order()
        bid.id = 99001
        bid.side = ex.Side.Buy
        bid.price = 100_0000
        bid.quantity = 10
        bid.type = ex.OrderType.Limit
        bid.tif = ex.TimeInForce.GTC
        bid.timestamp = 0
        bid.filled_quantity = 0
        bid.stop_price = 0
        bid.peg_offset = 0
        bid.visible_quantity = 0
        bid.hidden_quantity = 0
        engine.submit(bid)

        ask = ex.Order()
        ask.id = 99002
        ask.side = ex.Side.Sell
        ask.price = 101_0000
        ask.quantity = 10
        ask.type = ex.OrderType.Limit
        ask.tif = ex.TimeInForce.GTC
        ask.timestamp = 0
        ask.filled_quantity = 0
        ask.stop_price = 0
        ask.peg_offset = 0
        ask.visible_quantity = 0
        ask.hidden_quantity = 0
        engine.submit(ask)

        mm = MarketMakerAgent(agent_id=0)
        orders = mm.on_market_data(engine, timestamp=1000)

        assert len(orders) >= 2, f"Expected at least 2 orders (bid+ask), got {len(orders)}"
        sides = {o.side for o in orders}
        assert ex.Side.Buy in sides
        assert ex.Side.Sell in sides

    def test_bid_below_ask(self):
        """MarketMaker bid price is always below ask price."""
        engine = ex.MatchingEngine()

        bid = ex.Order()
        bid.id = 99001
        bid.side = ex.Side.Buy
        bid.price = 100_0000
        bid.quantity = 10
        bid.type = ex.OrderType.Limit
        bid.tif = ex.TimeInForce.GTC
        bid.timestamp = 0
        bid.filled_quantity = 0
        bid.stop_price = 0
        bid.peg_offset = 0
        bid.visible_quantity = 0
        bid.hidden_quantity = 0
        engine.submit(bid)

        ask = ex.Order()
        ask.id = 99002
        ask.side = ex.Side.Sell
        ask.price = 101_0000
        ask.quantity = 10
        ask.type = ex.OrderType.Limit
        ask.tif = ex.TimeInForce.GTC
        ask.timestamp = 0
        ask.filled_quantity = 0
        ask.stop_price = 0
        ask.peg_offset = 0
        ask.visible_quantity = 0
        ask.hidden_quantity = 0
        engine.submit(ask)

        mm = MarketMakerAgent(agent_id=0)
        orders = mm.on_market_data(engine, timestamp=1000)

        buy_orders = [o for o in orders if o.side == ex.Side.Buy]
        sell_orders = [o for o in orders if o.side == ex.Side.Sell]
        if buy_orders and sell_orders:
            assert buy_orders[0].price < sell_orders[0].price

    def test_inventory_skew(self):
        """With positive inventory, bid should be lower (discourages more buying)."""
        engine = ex.MatchingEngine()

        bid = ex.Order()
        bid.id = 99001
        bid.side = ex.Side.Buy
        bid.price = 100_0000
        bid.quantity = 10
        bid.type = ex.OrderType.Limit
        bid.tif = ex.TimeInForce.GTC
        bid.timestamp = 0
        bid.filled_quantity = 0
        bid.stop_price = 0
        bid.peg_offset = 0
        bid.visible_quantity = 0
        bid.hidden_quantity = 0
        engine.submit(bid)

        ask = ex.Order()
        ask.id = 99002
        ask.side = ex.Side.Sell
        ask.price = 101_0000
        ask.quantity = 10
        ask.type = ex.OrderType.Limit
        ask.tif = ex.TimeInForce.GTC
        ask.timestamp = 0
        ask.filled_quantity = 0
        ask.stop_price = 0
        ask.peg_offset = 0
        ask.visible_quantity = 0
        ask.hidden_quantity = 0
        engine.submit(ask)

        mm_neutral = MarketMakerAgent(agent_id=0)
        orders_neutral = mm_neutral.on_market_data(engine, timestamp=1000)
        buy_neutral = [o for o in orders_neutral if o.side == ex.Side.Buy]

        mm_long = MarketMakerAgent(agent_id=1)
        mm_long.inventory = 50
        orders_long = mm_long.on_market_data(engine, timestamp=1000)
        buy_long = [o for o in orders_long if o.side == ex.Side.Buy]

        if buy_neutral and buy_long:
            assert buy_long[0].price <= buy_neutral[0].price

    def test_seeds_empty_book(self):
        """MarketMaker seeds the book when it is empty."""
        engine = ex.MatchingEngine()
        mm = MarketMakerAgent(agent_id=0)
        orders = mm.on_market_data(engine, timestamp=0)
        assert len(orders) >= 1

    def test_stops_buying_at_max_inventory(self):
        """MM stops placing bids when inventory hits max."""
        engine = ex.MatchingEngine()

        bid = ex.Order()
        bid.id = 99001
        bid.side = ex.Side.Buy
        bid.price = 100_0000
        bid.quantity = 10
        bid.type = ex.OrderType.Limit
        bid.tif = ex.TimeInForce.GTC
        bid.timestamp = 0
        bid.filled_quantity = 0
        bid.stop_price = 0
        bid.peg_offset = 0
        bid.visible_quantity = 0
        bid.hidden_quantity = 0
        engine.submit(bid)

        ask = ex.Order()
        ask.id = 99002
        ask.side = ex.Side.Sell
        ask.price = 101_0000
        ask.quantity = 10
        ask.type = ex.OrderType.Limit
        ask.tif = ex.TimeInForce.GTC
        ask.timestamp = 0
        ask.filled_quantity = 0
        ask.stop_price = 0
        ask.peg_offset = 0
        ask.visible_quantity = 0
        ask.hidden_quantity = 0
        engine.submit(ask)

        mm = MarketMakerAgent(agent_id=0, max_inventory=15)
        mm.inventory = 15  # At max
        orders = mm.on_market_data(engine, timestamp=1000)

        buy_orders = [o for o in orders if o.side == ex.Side.Buy]
        assert len(buy_orders) == 0, "MM should not bid when at max inventory"


class TestMarketMakerIntegration:
    """Integration tests: MarketMaker in simulation."""

    def test_participates_in_trading(self):
        """MarketMaker generates fills over a simulation."""
        mm = MarketMakerAgent(agent_id=0)
        randoms = [RandomAgent(agent_id=i, seed=i * 13) for i in range(1, 4)]

        agents = [mm] + randoms
        sim = SimulationLoop(agents, num_steps=5000)
        results = sim.run()

        assert results["fills"] > 0
        assert len(mm.fills) > 0, "Market maker should participate in trading"

    def test_manages_inventory(self):
        """MarketMaker keeps inventory bounded (doesn't accumulate unboundedly)."""
        mm = MarketMakerAgent(agent_id=0, max_inventory=15)
        randoms = [RandomAgent(agent_id=i, seed=i * 7) for i in range(1, 4)]

        agents = [mm] + randoms
        sim = SimulationLoop(agents, num_steps=5000)
        sim.run()

        # Inventory should be bounded (within 3x max due to unwind mechanism)
        assert abs(mm.inventory) < mm.max_inventory * 3, (
            f"Inventory {mm.inventory} should be bounded near max_inventory={mm.max_inventory}"
        )

    def test_quotes_competitively(self):
        """MarketMaker quotes at or near BBO."""
        engine = ex.MatchingEngine()

        bid = ex.Order()
        bid.id = 99001
        bid.side = ex.Side.Buy
        bid.price = 100_0000
        bid.quantity = 10
        bid.type = ex.OrderType.Limit
        bid.tif = ex.TimeInForce.GTC
        bid.timestamp = 0
        bid.filled_quantity = 0
        bid.stop_price = 0
        bid.peg_offset = 0
        bid.visible_quantity = 0
        bid.hidden_quantity = 0
        engine.submit(bid)

        ask = ex.Order()
        ask.id = 99002
        ask.side = ex.Side.Sell
        ask.price = 101_0000
        ask.quantity = 10
        ask.type = ex.OrderType.Limit
        ask.tif = ex.TimeInForce.GTC
        ask.timestamp = 0
        ask.filled_quantity = 0
        ask.stop_price = 0
        ask.peg_offset = 0
        ask.visible_quantity = 0
        ask.hidden_quantity = 0
        engine.submit(ask)

        mm = MarketMakerAgent(agent_id=0)
        orders = mm.on_market_data(engine, timestamp=1000)

        buy_orders = [o for o in orders if o.side == ex.Side.Buy]
        sell_orders = [o for o in orders if o.side == ex.Side.Sell]

        if buy_orders:
            assert buy_orders[0].price >= 100_0000 - 100, "Bid should be near BBO"
        if sell_orders:
            assert sell_orders[0].price <= 101_0000 + 100, "Ask should be near BBO"

    def test_runs_full_simulation_without_crash(self):
        """Full 10000-step simulation with MM and randoms completes."""
        mm = MarketMakerAgent(agent_id=0)
        randoms = [RandomAgent(agent_id=i, seed=i * 42) for i in range(1, 6)]

        agents = [mm] + randoms
        sim = SimulationLoop(agents, num_steps=10000)
        results = sim.run()

        assert results["steps"] == 10000
        assert results["fills"] > 0


class TestQuoteScale:
    """The AS parameters have to match the price scale, and nothing else checks it.

    gamma, sigma and k are in price units. The book is in integer ticks. Get them
    wrong and the agent still runs, it just quotes somewhere unreachable or
    computes a skew too small to move an integer price. Both failures are silent,
    so they get pinned here in ticks.
    """

    def _quotes(self, inventory=0, **kwargs):
        engine = _one_tick_book()
        mm = MarketMakerAgent(agent_id=0, **kwargs)
        mm.inventory = inventory
        orders = mm.on_market_data(engine, timestamp=1000)
        bids = [o.price for o in orders if o.side == ex.Side.Buy]
        asks = [o.price for o in orders if o.side == ex.Side.Sell]
        return bids, asks

    def test_half_spread_is_a_few_ticks(self):
        """Flat inventory quotes both sides within a few ticks of the mid."""
        bids, asks = self._quotes()
        assert bids and asks
        # Mid of a one-tick book at 100_0000/100_0001.
        assert 0 < 100_0000.5 - bids[0] <= 3, f"bid {bids[0]} too far from mid"
        assert 0 < asks[0] - 100_0000.5 <= 3, f"ask {asks[0]} too far from mid"

    def test_bid_and_ask_are_different_ticks(self):
        """Sub-tick half-spread must not collapse both quotes onto one price."""
        bids, asks = self._quotes()
        assert bids[0] < asks[0]

    def test_skew_moves_the_quote_by_at_least_a_tick(self):
        """The regression guard: a skew that rounds to zero is not a skew.

        The shipped defaults used to give a skew of 0.0003 ticks, which cannot
        move an integer price, so inventory had no effect on the quotes at all.
        """
        flat_bids, _ = self._quotes(inventory=0)
        long_bids, _ = self._quotes(inventory=10)
        assert long_bids and flat_bids
        assert flat_bids[0] - long_bids[0] >= 1, (
            f"skew moved the bid {flat_bids[0] - long_bids[0]} ticks, "
            "so inventory is not reaching the quotes"
        )

    def test_skew_stays_bounded_near_max_inventory(self):
        """Skew grows with inventory but stays in single-digit ticks."""
        flat_bids, _ = self._quotes(inventory=0)
        heavy_bids, _ = self._quotes(inventory=14)
        moved = flat_bids[0] - heavy_bids[0]
        assert 1 <= moved <= 6, f"skew of {moved} ticks at q=14 is off scale"

    def test_skew_is_signed(self):
        """Short inventory lifts the bid, long inventory drops it."""
        short_bids, _ = self._quotes(inventory=-10)
        flat_bids, _ = self._quotes(inventory=0)
        long_bids, _ = self._quotes(inventory=10)
        assert long_bids[0] < flat_bids[0] < short_bids[0]

    def test_dollar_scale_parameters_are_visibly_wrong(self):
        """Documents the failure this class exists to catch.

        gamma=0.1, sigma=0.002, k=1.5 are reasonable for a stock quoted in
        dollars and were the shipped defaults. Here they put the quote hundreds
        of ticks off the book, where it can never fill.
        """
        bids, _ = self._quotes(gamma=0.1, sigma=0.002, k=1.5)
        assert 100_0000.5 - bids[0] > 100
