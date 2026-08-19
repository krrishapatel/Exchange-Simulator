# Exchange Simulator

[![CI](https://github.com/krrishapatel/Exchange-Simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/krrishapatel/Exchange-Simulator/actions/workflows/ci.yml)

High-performance simulated exchange with a C++ matching engine, ML trading agents, and live visualization.

## Performance

Benchmarked on Apple Silicon (Release build):

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Order book add | ~61 ns | 16.5M ops/sec |
| Order book cancel | ~34 ns | 29.8M ops/sec |
| Limit order match | ~118 ns | 8.5M matches/sec |
| Market order sweep (5 levels) | ~1029 ns | 972K sweeps/sec |

All operations well under the 1μs target.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Dashboard (React)          │  RL Training (Gymnasium + PPO)    │
├─────────────────────────────┼───────────────────────────────────┤
│  Python Bindings (pybind11)                                     │
├─────────────────────────────────────────────────────────────────┤
│  Agent Framework            │  Data Generation (Hawkes/Replay)  │
├─────────────────────────────┴───────────────────────────────────┤
│  C++ Matching Engine (price-time priority, zero-alloc hot path) │
└─────────────────────────────────────────────────────────────────┘
```

- **C++ Matching Engine**: Price-time priority order book with IOC/FOK/iceberg/stop/pegged orders, opening/closing auctions, zero-allocation memory pool
- **Python Bindings**: pybind11 wrapper exposing the full engine API to Python
- **Agent Framework**: BaseAgent interface, RandomAgent, Avellaneda-Stoikov MarketMaker
- **RL Environment**: Gymnasium-compliant env for training trading agents with PPO/SAC
- **Synthetic Data**: Hawkes process order flow with pre-built scenarios (calm, volatile, flash crash)
- **Live Dashboard**: WebSocket server + React frontend with price chart, order book, trade feed, agent PnL

## Quick Start

```bash
# Build C++ engine + Python bindings
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Run all C++ tests (80 tests)
ctest --test-dir build --output-on-failure

# Run Python tests
PYTHONPATH=build/bindings:. python3 -m pytest tests/ -v

# Run the dashboard
pip install websockets
PYTHONPATH=build/bindings:. python3 dashboard/server/app.py &
cd dashboard/frontend && npm install && npm run dev
```

## Project Structure

```
exchange-simulator/
├── engine/
│   ├── src/
│   │   ├── types.hpp              # Order, Fill, Side, Price, TimeInForce
│   │   ├── memory_pool.hpp        # Fixed-size pool allocator
│   │   ├── order_book.hpp/cpp     # L2/L3 book, price-time priority
│   │   ├── matching_engine.hpp/cpp # Execution engine + auction phases
│   │   └── auction.hpp/cpp        # Opening/closing uncross
│   ├── tests/                     # Google Test suite (80 tests)
│   └── bench/                     # Google Benchmark suite
├── bindings/                      # pybind11 Python wrapper
├── agents/
│   ├── base.py                    # BaseAgent interface
│   ├── random_agent.py            # Noise trader
│   └── market_maker.py            # Avellaneda-Stoikov market maker
├── rl/
│   ├── trading_env.py             # Gymnasium environment
│   └── train_ppo.py               # PPO training script
├── data/
│   ├── hawkes.py                  # Hawkes process generator
│   ├── replay.py                  # Lobster L3 replay
│   └── scenarios.py               # Pre-built market scenarios
├── simulation/
│   └── loop.py                    # Multi-agent simulation driver
├── dashboard/
│   ├── server/app.py              # WebSocket server (100 steps/sec)
│   └── frontend/                  # React + Canvas visualization
└── tests/                         # Python test suite
```

## Order Types

| Type | Description |
|------|-------------|
| Limit (GTC) | Rests on book until filled or cancelled |
| Limit (IOC) | Fill immediately, cancel remainder |
| Limit (FOK) | Fill entire quantity or reject |
| Market | Execute at best available price |
| Iceberg | Shows only visible_quantity, auto-refills |
| Stop | Activates as market order when price crosses trigger |
| StopLimit | Activates as limit order at trigger price |
| Pegged | Tracks best bid/ask with configurable offset |

## Agents

| Agent | Strategy |
|-------|----------|
| RandomAgent | Uniform random orders around mid, 8% of them crossing, orders expire after 25 steps |
| MarketMakerAgent | Avellaneda-Stoikov quoting, inventory skew, aggressive unwind |
| RL Agent | PPO via Gymnasium env. Learns two-sided quoting and inventory skew, matches the heuristic |
| `rl/baselines.py` | Reference policies the trained agent is measured against |

## Benchmarks

```bash
./build/engine/bench/bench_order_book
./build/engine/bench/bench_matching
```

## RL Training

```bash
pip install ".[rl]"

# Basic PPO training
PYTHONPATH=build/bindings:. python3 -m rl.train_ppo --total-timesteps 1000000

# Self-play training (league-style opponent pool)
PYTHONPATH=build/bindings:. python3 rl/self_play.py --total-timesteps 500000 --pool-size 10

# Evaluate a trained model
PYTHONPATH=build/bindings:. python3 rl/evaluate.py --model models/ppo_trader.zip --episodes 100
```

Models save to `models/`, one row per episode to `logs/ppo_trading/monitor.csv`.
That file is the learning curve; TensorBoard is optional and skipped if it is not
installed. Training ends with a comparison against the heuristics in
`rl/baselines.py` on seeds the agent never trained on, because a reward curve
going up says nothing about whether the policy is any good. On the default
settings the curve runs from -678 to about +2 over 1000 episodes and flattens
after roughly episode 500.

The comparison is the point. `random` loses about 400 ticks an episode, since two
of the five actions cross the spread; `inventory_aware`, which is a sign test on
the position, makes about 18; and the trained agent matches it. Judge on
`equity`, never on `pnl`, which is cash flow and reads as a large loss for
anything holding stock at the end.

## Data Replay & Backtesting

```bash
# Replay Lobster L3 data through the engine
PYTHONPATH=build/bindings:. python3 -c "
from data import LobsterReplay, run_backtest
from agents import MarketMakerAgent
replay = LobsterReplay('path/to/lobster.csv')
result = run_backtest(replay, [MarketMakerAgent(0)], max_events=10000)
print(result.summary())
"
```

Supports Lobster L3 and Databento MBO formats.

## Multi-Asset

```python
import exchange_simulator as ex
engine = ex.MultiAssetEngine()
engine.submit(1, buy_order)   # Symbol 1
engine.submit(2, sell_order)  # Symbol 2 (isolated book)
```

## Status

- [x] Core types and memory pool
- [x] Order book (price-time priority)
- [x] Matching engine (limit + market)
- [x] IOC/FOK/Iceberg/Stop/Pegged orders
- [x] Auction phases (opening/closing uncross)
- [x] Python bindings (pybind11)
- [x] Agent framework (classical strategies)
- [x] Two-sided noise trader flow. `RandomAgent` took the mid as
      `(best_bid + best_ask) // 2`, which floors onto the bid, so a sell placed at
      the mid landed on the bid and traded while a buy placed at the mid sat below
      the ask and never could. Sides were still drawn 50/50, so the flow looked
      balanced: 4557 buys and 4444 sells submitted over 3000 steps, 696 crossings,
      every one of them a sell, and the price walked monotonically down. Crossing
      is now an explicit symmetric probability, defaulted to the 7.7% rate the
      rounding accident produced. Any measurement taken before this is worthless.
- [x] RL reward on marked equity. Reward was the change in cash, which charges
      the full notional for a buy and credits it for a sell, so it ranked fixed
      policies close to backwards: over 500 steps always-buy-market scored
      -500,005,283 while being the most profitable policy at +31,592 marked, and
      always-sell-market scored +499,927,312 for half that profit. Anything
      trained on it learned to liquidate. Reward is now the change in cash plus
      inventory valued at the mid, in both `TradingEnv` and `SelfPlayEnv`.
- [x] Round-robin tournament with Elo. Matches run through `SelfPlayEnv`, so both
      agents compete for the same fills on one engine and are scored on
      marked-to-market PnL. This used to draw both players' PnL from
      `rng.normal(0, 100)`, so `python -m rl.analysis.tournament` printed a
      confident Elo ranking of your checkpoints that ignored the checkpoints. It
      had no tests, which is why nobody noticed.
- [x] Resting order expiry. `RandomAgent` never cancelled. Three of them add
      about 16 units of size per step and only ~8% of flow crosses, so the book
      thickened without bound until the touch was a wall nobody could clear: over
      3000 steps the touch reached 3258 units and the mid took 13 distinct values,
      moving on 0.4% of steps. There was no price discovery, and every number
      measured against that book was an artifact, including the volatility used to
      calibrate the market maker, which was 13 jumps averaged over 2987 frozen
      steps. Orders now expire after `order_lifetime` steps. A second defect was
      hiding behind this one: passive orders rested on a 100-unit grid on a book
      one unit wide, so clearing a level jumped the mid ~50 units and per-step
      volatility ran at 3x the spread. Both are fixed, and the resulting market
      has stationary depth, moves on 21% of steps, and a per-step volatility
      (0.48 ticks) below its spread (2.03) as a real book does.
- [x] Avellaneda-Stoikov market maker. Quotes at the reservation price plus and
      minus half the optimal spread, with an aggressive unwind past
      `max_inventory`. gamma, sigma and k are calibrated to this price scale,
      since the formulas are in price units and the book is in ticks. Over 10
      seeds of 3000 steps against a plain touch-quoting heuristic it returns
      394.6 +/- 161 marked to market against 388.7 +/- 152, winning in 4 of 10
      seeds, while carrying 2.5x less inventory (mean absolute position 3.3
      against 8.2, peak 13.9 against 18.8). It earns the same money with less
      risk, which is what the model is for: the skew is a risk control, not an
      alpha source, and against uninformed flow there is no adverse selection for
      it to avoid. An earlier version of this entry reported -69 against +887 and
      a 0-for-10 loss. Those numbers were measured in the frozen market described
      above and are withdrawn.
- [x] Order book imbalance features for ML. `OrderBook` gained `l2_bids` and
      `l2_asks`, returning per-level price, quantity and order count, so the RL
      observation now carries the real quantity resting at each of the top three
      levels on each side and a touch imbalance computed from quantity. It used
      to call `bid_depth()`, which counts price levels rather than quantity, and
      split that one number across three "levels" on fixed 3:2:1 weights, so
      levels 1 and 2 held no information the sum did not already have and the
      imbalance moved with how spread out the book was rather than with pressure.
- [x] Deep RL self-play convergence analysis. PPO on `TradingEnv` for 1M steps
      converges, and it converges to the hand-written heuristic rather than past
      it. Over 400 held-out seeds the trained policy returns 17.5 +/- 0.4 against
      17.9 +/- 0.5 for a six-line inventory-flattening rule: a paired difference
      of -0.1 +/- 0.4, ahead in exactly 200 of 400 seeds. It gets there on its
      own, which is the interesting part. Sweeping only the position feature flips
      its action exactly at zero, from `buy_limit` when flat or short one to
      `sell_limit` at plus one and above, escalating to `buy_market` past three
      short, and in a live episode it quotes the flattening side on 94.3% of the
      steps where it trades while holding a position. That is the shape of the
      Avellaneda-Stoikov skew plus threshold unwind, learned from the reward
      alone. It never crosses the spread unprompted. Reproduce with
      `python -m rl.train_ppo`, and see `rl/baselines.py` for the reference
      policies. Two caveats: the inventory penalty is doing real work, since at
      the old default of 0.005 the same recipe learned a policy holding 7.7 units
      that did not beat the heuristic (-3.0 +/- 5.5 over 400 seeds); and this is
      single-agent training, so league-style self-play convergence is still
      unmeasured.
- [x] Gymnasium RL environment
- [x] Self-play RL training (league-style)
- [x] Synthetic data generator (Hawkes process)
- [x] Real data replay (Lobster/Databento)
- [x] Backtesting harness
- [x] Live dashboard (WebSocket + React)
- [x] Latency histogram panel
- [x] Multi-asset matching engine
- [x] FIX protocol gateway. Parser and session layer, 14 tests. The four TCP
      integration tests are skipped and hang if forced, so the transport itself
      is not verified.
- [ ] League-style self-play convergence. `SelfPlayEnv` runs real matches and
      both players now requote and share one position limit, but the convergence
      result above is single-agent. No Elo curve has been produced from a league
      of trained checkpoints playing each other.

## License

MIT
