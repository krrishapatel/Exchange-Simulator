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
│   └── market_maker.py            # Avellaneda-Stoikov MM
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
| RandomAgent | Uniform random orders around mid (noise) |
| MarketMakerAgent | Avellaneda-Stoikov optimal quoting with inventory skew |
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
- [x] Avellaneda-Stoikov market maker
- [x] Gymnasium RL environment
- [x] Self-play RL training (league-style)
- [x] Synthetic data generator (Hawkes process)
- [x] Real data replay (Lobster/Databento)
- [x] Backtesting harness
- [x] Live dashboard (WebSocket + React)
- [x] Latency histogram panel
- [x] Multi-asset matching engine
- [ ] Deep RL self-play convergence analysis
- [ ] FIX protocol gateway
- [ ] Order book imbalance features for ML

## License

MIT
