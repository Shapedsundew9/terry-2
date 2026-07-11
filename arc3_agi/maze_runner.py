"""Batch parallel runner for Maze populations.

Launches :data:`NUM_POPULATIONS` independent maze populations concurrently
using :mod:`arc3_agi.runner`, reports live progress to the terminal, and
blocks until all populations have completed :data:`MAX_GENERATIONS` generations.

All tuneable parameters are defined as module-level constants so they can be
changed without digging into the code.

Run directly::

    .venv/bin/python -m arc3_agi.maze_runner

or import and call :func:`run` from another script.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

from arc3_agi.checkpoint import CheckpointConfig
from arc3_agi.fingerprint import FingerprintConfig
from arc3_agi.maze import Maze, MazeAutomaton
from arc3_agi.runner import (
    PopulationConfig,
    PopulationHandle,
    launch_populations,
    wait_all,
)

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

NUM_POPULATIONS: int = 14
"""Number of independent populations to evolve in parallel."""

MAX_GENERATIONS: int = 10000
"""Total number of tick/evolve cycles each population runs before stopping."""

TICKS_PER_RESTART: int = 100
"""Number of :meth:`~arc3_agi.population.Population.tick` calls per restart."""

RESTARTS_PER_GEN: int = 20
"""Number of independent restarts per generation.  Fitness is averaged across
all restarts to reduce starting-condition bias.  Set to 1 to reproduce the
original single-attempt behaviour.
"""

POPULATION_SIZE: int = 100
"""Number of automata in each population."""

SIDE_LENGTH_BITS: int = 6
"""Maze grid side length is ``2 ** SIDE_LENGTH_BITS`` (6 → 64×64)."""

MAZE_SEED: int = 42
"""Shared seed for deterministic maze generation (same maze for all populations)."""

POPULATION_SEED: int = 0
"""Base seed for deterministic evolution.  Each population i receives seed
``POPULATION_SEED + i`` so populations are independent yet fully reproducible.
Set to ``None`` to use OS entropy (non-deterministic).
"""

FINGERPRINT_BITS: int = 4
"""Bit-width of the selection fingerprint."""

FINGERPRINT_TOURNAMENT_K: int = 4
"""Tournament size for fingerprint-guided mate selection."""

CHECKPOINT_INTERVAL: int = 1000
"""Write a checkpoint every this many generations (0 = disable)."""

POLL_INTERVAL_S: float = 2.0
"""Seconds between progress-table refreshes while populations are running."""

BASE_DIR: Path = Path("runs")
"""Root directory under which per-run checkpoint folders are created."""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# ANSI escape codes — used only when stdout is an interactive TTY.
_CSI = "\033["
_CURSOR_UP = _CSI + "{}A"  # move cursor up N lines
_ERASE_LINE = _CSI + "2K\r"  # erase entire current line


def _is_tty() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _format_table(
    handles: list[PopulationHandle],
    max_generations: int,
    elapsed_s: float,
) -> list[str]:
    """Return a list of formatted status lines (one per population + header)."""
    done = sum(1 for h in handles if not h.is_running)
    lines: list[str] = [
        f"  Elapsed: {elapsed_s:6.0f}s   Finished: {done}/{len(handles)} populations",
        f"  {'Pop':>3}  {'Gen':>6}/{max_generations:<6}  {'Max fit':>9}  {'Mean fit':>9}  "
        f"{'Best Max':>9}  {'Best Mean':>9}  {'Status':<8}",
        "  " + "-" * 78,
    ]

    # Accumulators for the summary row.
    sum_max = sum_mean = sum_best_max = sum_best_mean = 0.0
    n_valid = 0

    # Cache progress snapshots so we don't drain the queue twice.
    snapshots: list[dict] = []
    for h in handles:
        snapshots.append(h.progress)

    for h, prog in zip(handles, snapshots):
        gen = prog.get("generation", 0)
        mx = prog.get("max_fitness", float("nan"))
        mn = prog.get("mean_fitness", float("nan"))
        bm = prog.get("best_max_fitness", float("nan"))
        bmn = prog.get("best_mean_fitness", float("nan"))
        status = "done" if not h.is_running else "running"
        pct = 100 * gen / max_generations if max_generations else 0
        bar_filled = int(pct / 10)
        bar = "[" + "#" * bar_filled + "." * (10 - bar_filled) + "]"
        lines.append(
            f"  {h.population_id:>3}  {gen:>6}/{max_generations:<6}  "
            f"{mx:>9.3f}  {mn:>9.3f}  {bm:>9.3f}  {bmn:>9.3f}  "
            f"{status:<8}  {pct:5.1f}% {bar}"
        )
        if (
            not math.isnan(mx)
            and not math.isnan(mn)
            and not math.isnan(bm)
            and not math.isnan(bmn)
        ):
            sum_max += mx
            sum_mean += mn
            sum_best_max += bm
            sum_best_mean += bmn
            n_valid += 1

    # Summary row.
    lines.append("  " + "-" * 78)
    if n_valid:
        avg_max = sum_max / n_valid
        avg_mean = sum_mean / n_valid
        avg_best_max = sum_best_max / n_valid
        avg_best_mean = sum_best_mean / n_valid
        lines.append(
            f"  {'AVG':>3}  {'':>6} {'':6}  "
            f"{avg_max:>9.3f}  {avg_mean:>9.3f}  {avg_best_max:>9.3f}  {avg_best_mean:>9.3f}  "
            f"{'':8}"
        )
    else:
        lines.append(f"  {'AVG':>3}  (no data yet)")
    return lines


def _print_progress(
    handles: list[PopulationHandle],
    max_generations: int,
    elapsed_s: float,
    *,
    first: bool = False,
    tty: bool,
    prev_lines: int = 0,
) -> int:
    """Render the progress table, overwriting previous output on a TTY.

    Returns the number of lines printed (so the next call can erase them).
    """
    lines = _format_table(handles, max_generations, elapsed_s)
    if tty and not first and prev_lines:
        # Move cursor up to the start of the previous block and erase each line.
        sys.stdout.write(_CURSOR_UP.format(prev_lines))
        for _ in range(prev_lines):
            sys.stdout.write(_ERASE_LINE + "\n")
        sys.stdout.write(_CURSOR_UP.format(prev_lines))
    for line in lines:
        sys.stdout.write(line + "\n")
    sys.stdout.flush()
    return len(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_configs(maze: Maze) -> list[PopulationConfig]:
    """Build :data:`NUM_POPULATIONS` identical :class:`~arc3_agi.runner.PopulationConfig` objects.

    All populations share the same maze instance and hyperparameters.
    Each runs in its own subprocess, so the shared Python object is forked
    into isolated memory — no cross-population state leaks.

    Parameters
    ----------
    maze:
        The :class:`~arc3_agi.maze.Maze` environment to use for all populations.

    Returns
    -------
    list[PopulationConfig]
        ``NUM_POPULATIONS`` configs ready to pass to :func:`~arc3_agi.runner.launch_populations`.
    """
    fp_cfg = FingerprintConfig(
        bits=FINGERPRINT_BITS, tournament_k=FINGERPRINT_TOURNAMENT_K
    )
    ckpt_cfg = CheckpointConfig(
        enabled=CHECKPOINT_INTERVAL > 0,
        generation_interval=CHECKPOINT_INTERVAL,
    )
    return [
        PopulationConfig(
            size=POPULATION_SIZE,
            AutomatonClass=MazeAutomaton,
            environment=maze,
            ticks_per_restart=TICKS_PER_RESTART,
            restarts_per_gen=RESTARTS_PER_GEN,
            checkpoint_config=ckpt_cfg,
            fingerprint_config=None,  # fp_cfg,
            seed=POPULATION_SEED + i if POPULATION_SEED is not None else None,
        )
        for i in range(NUM_POPULATIONS)
    ]


def run(base_dir: Path = BASE_DIR) -> list[PopulationHandle]:
    """Launch all populations, report live progress, and wait for completion.

    Parameters
    ----------
    base_dir:
        Root directory for checkpoint output.  Defaults to :data:`BASE_DIR`.

    Returns
    -------
    list[PopulationHandle]
        One handle per population; all are finished when this function returns.
    """
    tty = _is_tty()

    maze = Maze(
        name="MazeRunnerMaze", side_length_bits=SIDE_LENGTH_BITS, seed=MAZE_SEED
    )
    configs = build_configs(maze)

    print(
        f"\nMaze Runner — {NUM_POPULATIONS} populations × {MAX_GENERATIONS} generations "
        f"× {TICKS_PER_RESTART} ticks/restart × {RESTARTS_PER_GEN} restart(s)/gen\n"
        f"  Maze: {maze.width}×{maze.height}  "
        f"Population size: {POPULATION_SIZE}  "
        f"Checkpoint every: {CHECKPOINT_INTERVAL} gens\n"
        f"  Checkpoints → {base_dir.resolve()}\n"
    )

    handles = launch_populations(
        configs, max_generations=MAX_GENERATIONS, base_dir=base_dir
    )
    t0 = time.monotonic()

    prev_lines = _print_progress(
        handles, MAX_GENERATIONS, elapsed_s=0.0, first=True, tty=tty
    )

    while any(h.is_running for h in handles):
        time.sleep(POLL_INTERVAL_S)
        prev_lines = _print_progress(
            handles,
            MAX_GENERATIONS,
            elapsed_s=time.monotonic() - t0,
            first=False,
            tty=tty,
            prev_lines=prev_lines,
        )

    # Final update after all processes have exited.
    _print_progress(
        handles,
        MAX_GENERATIONS,
        elapsed_s=time.monotonic() - t0,
        first=False,
        tty=tty,
        prev_lines=prev_lines,
    )

    total_s = time.monotonic() - t0
    print(f"\nAll {NUM_POPULATIONS} populations finished in {total_s:.1f}s.")
    return handles


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run()
