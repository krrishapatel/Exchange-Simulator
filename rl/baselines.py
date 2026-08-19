"""Heuristic policies for TradingEnv, used as the bar a trained agent must clear.

A learning curve on its own says nothing about whether a policy is any good. It
shows reward going up, but reward going up from what to what? These policies are
the reference points, all measured in the same env with the same action space, so
"the agent learned something" becomes a comparison rather than a claim.

Measured over 20 seeds of 1000 steps on the default market:

    policy                    equity        mean|inv|   max|inv|
    hold                        0.0 +/-  0.0      0.0        0.0
    random                   -420.6 +/- 20.4     11.0       21.9
    always_buy_limit           52.6 +/- 32.3     18.3       25.0
    always_sell_limit          -4.2 +/- 31.7     18.7       25.0
    alternate_quote            25.4 +/- 11.4      4.2        8.8
    inventory_aware            20.1 +/-  2.4      0.5        1.0
    lean_on_imbalance          14.6 +/-  1.8      0.9        3.0

``inventory_aware`` is the one to beat. It earns the least of the profitable
policies bar one, but it earns it with a t-stat around 8 and a mean absolute
position of half a unit, so the profit is the spread rather than a directional
bet that happened to land. ``always_buy_limit`` posts a bigger number with a
t-stat of 1.6 and its position pinned at the limit all episode: that is a punt on
the mid, and across 40 seeds the mid drifts up in 22 and down in 17, so there is
no edge there to keep.

``random`` losing 420 is the sanity check. Two of the five actions cross the
spread, so anything that acts without a reason pays for it, and a policy that has
learned nothing at all should land near this number rather than near zero.
"""

from __future__ import annotations

import numpy as np

# Action indices, matching TradingEnv.
HOLD = 0
BUY_LIMIT = 1
BUY_MARKET = 2
SELL_LIMIT = 3
SELL_MARKET = 4


class Baseline:
    """A fixed policy. Takes the observation and the env's info dict, returns an action.

    Both are passed because some of these need the position, which is in info
    rather than being read back out of the normalized observation.
    """

    name = "baseline"

    def reset(self) -> None:
        """Clear any per-episode state."""

    def act(self, obs: np.ndarray, info: dict) -> int:
        raise NotImplementedError


class Hold(Baseline):
    """Never trade. Scores exactly zero, so it separates profit from noise."""

    name = "hold"

    def act(self, obs: np.ndarray, info: dict) -> int:
        return HOLD


class Random(Baseline):
    """Uniform over all five actions. The floor, not a strategy."""

    name = "random"

    def __init__(self, seed: int = 0):
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)

    def act(self, obs: np.ndarray, info: dict) -> int:
        return int(self._rng.integers(0, 5))


class AlwaysBuyLimit(Baseline):
    """Bid at the touch every step. Ends the episode pinned at the position limit."""

    name = "always_buy_limit"

    def act(self, obs: np.ndarray, info: dict) -> int:
        return BUY_LIMIT


class AlwaysSellLimit(Baseline):
    """Offer at the touch every step. The mirror of AlwaysBuyLimit."""

    name = "always_sell_limit"

    def act(self, obs: np.ndarray, info: dict) -> int:
        return SELL_LIMIT


class AlternateQuote(Baseline):
    """Alternate bid and offer at the touch.

    The action space allows one order per step, so this is the closest thing to a
    two-sided quote available. It earns the spread but drifts, because which side
    fills is not under its control.
    """

    name = "alternate_quote"

    def __init__(self):
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def act(self, obs: np.ndarray, info: dict) -> int:
        self._step += 1
        return BUY_LIMIT if self._step % 2 else SELL_LIMIT


class InventoryAware(Baseline):
    """Quote the side that flattens the position, alternating when flat.

    This is AlternateQuote with the drift removed: after a buy fills it offers
    rather than bidding again, so the position oscillates around zero instead of
    wandering. Same idea as the inventory skew in Avellaneda-Stoikov, done with a
    sign test instead of a reservation price.
    """

    name = "inventory_aware"

    def __init__(self):
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def act(self, obs: np.ndarray, info: dict) -> int:
        self._step += 1
        inventory = info.get("inventory", 0)
        if inventory > 0:
            return SELL_LIMIT
        if inventory < 0:
            return BUY_LIMIT
        return BUY_LIMIT if self._step % 2 else SELL_LIMIT


class LeanOnImbalance(Baseline):
    """Take the side the touch is leaning towards, else flatten.

    Uses obs[10], the quantity imbalance at the touch. Included because it is the
    simplest possible use of the feature the roadmap asked for, so it shows
    whether that feature carries anything a position-only rule does not already
    have. It does not: this earns less than InventoryAware.
    """

    name = "lean_on_imbalance"

    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold
        self._fallback = InventoryAware()

    def reset(self) -> None:
        self._fallback.reset()

    def act(self, obs: np.ndarray, info: dict) -> int:
        if obs[10] > self.threshold:
            return BUY_LIMIT
        if obs[10] < -self.threshold:
            return SELL_LIMIT
        return self._fallback.act(obs, info)


def all_baselines() -> list[Baseline]:
    """Every baseline, in the order the docstring table reports them."""
    return [
        Hold(),
        Random(),
        AlwaysBuyLimit(),
        AlwaysSellLimit(),
        AlternateQuote(),
        InventoryAware(),
        LeanOnImbalance(),
    ]


def run_baseline(policy: Baseline, env, seed: int) -> dict:
    """Run one policy for one episode, returning equity and position stats.

    Judge on ``equity``. ``pnl`` is cash flow, so it reads as a large loss for
    anything holding stock at the end.
    """
    policy.reset()
    obs, info = env.reset(seed=seed)
    inventories = []
    terminated = truncated = False
    while not (terminated or truncated):
        obs, _, terminated, truncated, info = env.step(policy.act(obs, info))
        inventories.append(info["inventory"])

    return {
        "policy": policy.name,
        "seed": seed,
        "equity": info["equity"],
        "final_inventory": info["inventory"],
        "mean_abs_inventory": float(np.mean(np.abs(inventories))),
        "max_abs_inventory": int(np.max(np.abs(inventories))),
    }
