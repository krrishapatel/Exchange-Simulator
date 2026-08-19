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

# Run all C++ tests (66 tests)
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
│   ├── tests/                     # Google Test suite (66 tests)
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
| RandomAgent | Uniform random orders around mid, 8% of them crossing (noise) |
| MarketMakerAgent | Avellaneda-Stoikov quoting, inventory skew, aggressive unwind |
| RL Agent | PPO-trained via Gymnasium environment |

## Benchmarks

```bash
./build/engine/bench/bench_order_book
./build/engine/bench/bench_matching
```

## RL Training

```bash
pip install ".[rl]"

# Basic PPO training
PYTHONPATH=build/bindings:. python3 rl/train_ppo.py --timesteps 100000

# Self-play training (league-style opponent pool)
PYTHONPATH=build/bindings:. python3 rl/self_play.py --timesteps 500000 --pool-size 10

# Evaluate a trained model
PYTHONPATH=build/bindings:. python3 rl/evaluate.py --model models/ppo_trader.zip --episodes 100
```

Models save to `models/`. TensorBoard logs to `logs/`.

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
- [x] Avellaneda-Stoikov market maker. Quotes at the reservation price plus and
      minus half the optimal spread, with an aggressive unwind past
      `max_inventory`. gamma, sigma and k are calibrated to this price scale from
      the measured per-step mid volatility (0.000234) and book spread (one unit in
      99% of steps), since the formulas are in price units and the book is in
      ticks. It is a correct implementation of the model and it is not the best
      quoter here: over 10 seeds against a plain touch-quoting heuristic it takes
      9.1 fills per run against 10.7 and returns -69 marked to market against
      +887, worse in 10 out of 10 seeds. 89% of its fills are passive. The model
      assumes order arrival intensity decaying exponentially in the distance from
      the mid, and `RandomAgent` crosses at a fixed probability whatever the
      distance, so the spread it computes is optimal for a different market.
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
- [ ] Deep RL self-play convergence analysis. The tracker, the Elo table and the
      tournament all run real matches now, but no convergence result has been
      produced from trained checkpoints, and none of the numbers above involve a
      trained policy.
- [ ] Order book imbalance features for ML

## License

MIT
