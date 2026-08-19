"""League convergence report: do later self-play checkpoints beat earlier ones?

`tournament.py` runs the all-pairs matches and gives a sequential Elo, which is
what you rank a live pool with. This asks a narrower question that a single Elo
ordering cannot answer on its own: as training goes on, does the policy get
stronger? That is the whole claim behind "self-play converges", and it needs a
statistic that does not depend on the order matches happen to be scored in.

So the numbers here are order-independent. Each generation's score is the
fraction of episodes it wins across the whole round-robin, which does not care
about match order at all, and the Elo is averaged over many random orderings so
the leftover order sensitivity is visible as a spread rather than hidden in one
number. The convergence signal is the Spearman correlation between training step
and score: positive means later is stronger, near zero means the extra training
bought no skill, negative means it regressed.

A `random` anchor is included by default. It exists to prove the tournament can
tell policies apart at all: if it does not come last by a wide margin, a flat
result among the trained generations says nothing.
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path

import numpy as np

from rl.analysis.elo import EloRating
from rl.analysis.tournament import AgentEntry, simulate_match


def _spearman(x, y) -> float:
    """Rank correlation, so a monotonic but non-linear trend still reads as 1."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    # A constant column has no ranking to correlate. Checked on the raw values,
    # since argsort hands ties distinct ranks and would hide the degenerate case.
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def run_league(
    players: list[AgentEntry],
    steps: list[int] | None = None,
    *,
    episodes_per_match: int = 30,
    episode_length: int = 1000,
    seed: int = 1000,
    elo_orderings: int = 200,
    sim_fn=simulate_match,
) -> dict:
    """Play an all-pairs round-robin and summarise it order-independently.

    Args:
        players: The agents to rank. Generation checkpoints, oldest first, plus
            any anchors.
        steps: Training step for each player, used only to correlate with score.
            None for anchors that have no step. Length must match ``players``.
        episodes_per_match: Episodes per matchup. Every matchup uses paired seeds,
            so both agents in it face the same episodes.
        episode_length: Steps per episode.
        seed: Base seed. Distinct per matchup so matchups are independent.
        elo_orderings: How many random match orderings to average Elo over. The
            spread across them is the order sensitivity, reported as a sd.
        sim_fn: Match runner, injectable so tests do not need a trained model.

    Returns:
        Dict with per-player ``avg_score``, ``elo`` (mean and sd), the ``win``
        matrix (row beats col), and the step/score and step/Elo Spearman.
    """
    names = [p.name for p in players]
    n = len(players)
    if steps is not None and len(steps) != n:
        raise ValueError("steps must line up with players")

    win = np.full((n, n), np.nan)
    score = np.zeros(n)
    games = np.zeros(n)
    scored: list[tuple[str, str, float]] = []

    for i, j in itertools.combinations(range(n), 2):
        results = sim_fn(
            players[i], players[j], episodes_per_match, episode_length,
            seed + (i * n + j) * 100,
        )
        if not results:
            continue
        wi = sum(1 for r in results if r.winner == players[i].name)
        wj = sum(1 for r in results if r.winner == players[j].name)
        draws = len(results) - wi - wj
        win[i][j] = wi / len(results)
        win[j][i] = wj / len(results)
        score[i] += wi + 0.5 * draws
        score[j] += wj + 0.5 * draws
        games[i] += len(results)
        games[j] += len(results)
        scored += [(players[i].name, players[j].name, r.result_for_a) for r in results]

    avg_score = {names[k]: (score[k] / games[k] if games[k] else float("nan")) for k in range(n)}

    # Elo, averaged over random orderings so no single order is privileged.
    acc: dict[str, list[float]] = {nm: [] for nm in names}
    for s in range(max(1, elo_orderings)):
        order = list(scored)
        random.Random(s).shuffle(order)
        elo = EloRating()
        for a, b, res in order:
            elo.update(a, b, res)
        for nm, v in elo.ratings().items():
            acc[nm].append(v)
    elo_mean = {nm: float(np.mean(v)) if v else float("nan") for nm, v in acc.items()}
    elo_sd = {nm: float(np.std(v)) if v else float("nan") for nm, v in acc.items()}

    # Correlate training progress with skill, over the players that have a step.
    trend_steps, trend_score, trend_elo = [], [], []
    for k in range(n):
        if steps is not None and steps[k] is not None:
            trend_steps.append(steps[k])
            trend_score.append(avg_score[names[k]])
            trend_elo.append(elo_mean[names[k]])

    return {
        "names": names,
        "steps": steps,
        "avg_score": avg_score,
        "elo_mean": elo_mean,
        "elo_sd": elo_sd,
        "win": win,
        "spearman_step_score": _spearman(trend_steps, trend_score),
        "spearman_step_elo": _spearman(trend_steps, trend_elo),
        "episodes_per_match": episodes_per_match,
    }


def format_report(report: dict) -> str:
    """A plain-text convergence report, sorted by score."""
    names = report["names"]
    avg = report["avg_score"]
    elo = report["elo_mean"]
    sd = report["elo_sd"]
    lines = [
        f"League of {len(names)} agents, {report['episodes_per_match']} episodes/matchup",
        "",
        f"{'agent':>10}  {'score':>6}  {'elo':>6}",
    ]
    for nm in sorted(names, key=lambda x: -avg[x]):
        lines.append(f"{nm:>10}  {avg[nm]:6.3f}  {elo[nm]:4.0f}+/-{sd[nm]:.0f}")
    ss = report["spearman_step_score"]
    se = report["spearman_step_elo"]
    lines += [
        "",
        f"Spearman(training step, score) = {ss:+.3f}",
        f"Spearman(training step, elo)   = {se:+.3f}",
        "",
        "Positive means later checkpoints are stronger. Near zero means the extra",
        "training bought no skill against the population.",
    ]
    return "\n".join(lines)


def _load_pool(pool_dir: Path, snapshot_interval: int | None) -> tuple[list[AgentEntry], list[int | None]]:
    """Load gen_*.zip checkpoints oldest first, labelling each with its step."""
    players, steps = [], []
    checkpoints = sorted(pool_dir.glob("gen_*.zip"))
    if not checkpoints:
        return [], []
    for idx, path in enumerate(checkpoints):
        players.append(AgentEntry(name=path.stem, checkpoint_path=path))
        # gen_0001 is the initial snapshot at step 0, gen_0002 the first
        # interval, and so on. Without an interval, label by index instead.
        steps.append(idx * snapshot_interval if snapshot_interval else idx)
    return players, steps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m rl.analysis.league",
        description="Report whether later self-play checkpoints beat earlier ones.",
    )
    ap.add_argument("--pool-dir", default="models/opponent_pool")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--episode-length", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument(
        "--snapshot-interval",
        type=int,
        default=None,
        help="Steps between checkpoints, to label the x-axis. Defaults to index.",
    )
    ap.add_argument(
        "--no-random-anchor",
        action="store_true",
        help="Drop the random baseline that proves the tournament discriminates.",
    )
    args = ap.parse_args(argv)

    players, steps = _load_pool(Path(args.pool_dir), args.snapshot_interval)
    if len(players) < 2:
        print(f"need at least 2 checkpoints in {args.pool_dir}", file=sys.stderr)
        return 2

    if not args.no_random_anchor:
        players.append(
            AgentEntry(name="random", action_fn=lambda obs: random.Random().randint(0, 4))
        )
        steps.append(None)

    report = run_league(
        players, steps,
        episodes_per_match=args.episodes,
        episode_length=args.episode_length,
        seed=args.seed,
    )
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
