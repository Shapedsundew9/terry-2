"""Batch parallel runner for Maze populations.

Launches up to :data:`MAX_PARALLEL` independent maze populations concurrently
using :mod:`arc3_agi.runner`, reports live progress to the terminal, and
blocks until all populations have completed :data:`MAX_GENERATIONS` generations.

Two entry points are available:

* :func:`run` — launches exactly :data:`MAX_PARALLEL` populations and waits
  for all of them to finish.
* :func:`run_pool` — launches :data:`TOTAL_POPULATIONS` populations in total,
  keeping at most :data:`MAX_PARALLEL` running concurrently and replacing each
  finished population with a new one until the quota is met.

All tuneable parameters are defined as module-level constants so they can be
changed without digging into the code.

Run directly::

    .venv/bin/python -m arc3_agi.maze_runner

or import and call :func:`run` or :func:`run_pool` from another script.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any

from arc3_agi.checkpoint import CheckpointConfig
from arc3_agi.experiment import ExperimentStore
from arc3_agi.fingerprint import FingerprintConfig
from arc3_agi.maze import Maze, MazeAutomaton
from arc3_agi.runner import (
    PopulationConfig,
    PopulationHandle,
    launch_populations,
    stop_all,
    wait_all,
)

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

MAX_PARALLEL: int = 12
"""Maximum number of populations to evolve concurrently."""

TOTAL_POPULATIONS: int = 100
"""Total number of populations to run across all batches (pool mode).

When :func:`run_pool` is used, populations are launched and replaced until
exactly this many have completed.  Has no effect on :func:`run`.
"""

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

FINGERPRINT_ENABLED: bool = False
"""Whether to enable fingerprint-guided mate selection."""

FINGERPRINT_BITS: int = 4
"""Bit-width of the selection fingerprint."""

FINGERPRINT_TOURNAMENT_K: int = 4
"""Tournament size for fingerprint-guided mate selection."""

FINGERPRINT_MUTATION_RATE: float = 0.01
"""Per-bit mutation probability for inherited selection fingerprints."""

AUTOMATON_PARAMS: dict[str, Any] = {"state_bits": 4}
"""Keyword arguments forwarded to :class:`~arc3_agi.maze.MazeAutomaton`."""

CHECKPOINT_INTERVAL: int = MAX_GENERATIONS
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
    completed_snapshots: list[dict] | None = None,
    total_populations: int | None = None,
) -> list[str]:
    """Return a list of formatted status lines (one per population + header)."""
    done = sum(1 for h in handles if not h.is_running)
    if completed_snapshots is None:
        header = f"  Elapsed: {elapsed_s:6.0f}s   Finished: {done}/{len(handles)} populations"
    else:
        total = (
            total_populations if total_populations is not None else TOTAL_POPULATIONS
        )
        header = (
            f"  Elapsed: {elapsed_s:6.0f}s   Active: {done}/{len(handles)} done"
            f"   Completed: {len(completed_snapshots)}/{total} total"
        )
    lines: list[str] = [
        header,
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
    total = total_populations if total_populations is not None else TOTAL_POPULATIONS
    if completed_snapshots:
        c_bm = [s.get("best_max_fitness", float("nan")) for s in completed_snapshots]
        c_bmn = [s.get("best_mean_fitness", float("nan")) for s in completed_snapshots]
        valid_bm = [v for v in c_bm if not math.isnan(v)]
        valid_bmn = [v for v in c_bmn if not math.isnan(v)]
        avg_c_bm = sum(valid_bm) / len(valid_bm) if valid_bm else float("nan")
        avg_c_bmn = sum(valid_bmn) / len(valid_bmn) if valid_bmn else float("nan")
        lines.append(
            f"  {'':>3}  Completed: {len(completed_snapshots)}/{total}"
            f"   avg best_max: {avg_c_bm:>9.3f}   avg best_mean: {avg_c_bmn:>9.3f}"
        )
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
    completed_snapshots: list[dict] | None = None,
    total_populations: int | None = None,
) -> int:
    """Render the progress table, overwriting previous output on a TTY.

    Returns the number of lines printed (so the next call can erase them).
    """
    lines = _format_table(
        handles,
        max_generations,
        elapsed_s,
        completed_snapshots,
        total_populations,
    )
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


def default_experiment_params() -> dict[str, Any]:
    """Return the default experiment parameters derived from module constants."""
    return {
        "total_populations": TOTAL_POPULATIONS,
        "max_parallel": MAX_PARALLEL,
        "max_generations": MAX_GENERATIONS,
        "ticks_per_restart": TICKS_PER_RESTART,
        "restarts_per_gen": RESTARTS_PER_GEN,
        "population_size": POPULATION_SIZE,
        "side_length_bits": SIDE_LENGTH_BITS,
        "maze_seed": MAZE_SEED,
        "population_seed": POPULATION_SEED,
        "fingerprint_enabled": FINGERPRINT_ENABLED,
        "fingerprint_bits": FINGERPRINT_BITS,
        "fingerprint_tournament_k": FINGERPRINT_TOURNAMENT_K,
        "fingerprint_mutation_rate": FINGERPRINT_MUTATION_RATE,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "poll_interval_s": POLL_INTERVAL_S,
        "automaton_params": dict(AUTOMATON_PARAMS),
    }


def _resolve_experiment_params(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge caller-supplied experiment params over the module defaults."""
    resolved = default_experiment_params()
    if not params:
        return resolved

    for key, value in params.items():
        if key == "automaton_params":
            automaton_params = dict(resolved["automaton_params"])
            if value is not None:
                automaton_params.update(value)
            resolved[key] = automaton_params
        else:
            resolved[key] = value
    return resolved


def _population_seed(pop_id: int, params: dict[str, Any]) -> int | None:
    seed = params.get("population_seed")
    return int(seed) + pop_id if seed is not None else None


def _fingerprint_config(params: dict[str, Any]) -> FingerprintConfig | None:
    if not params.get("fingerprint_enabled", False):
        return None
    return FingerprintConfig(
        bits=int(params["fingerprint_bits"]),
        tournament_k=int(params["fingerprint_tournament_k"]),
        mutation_rate=float(
            params.get("fingerprint_mutation_rate", FINGERPRINT_MUTATION_RATE)
        ),
    )


def _build_config(
    pop_id: int,
    maze: Maze,
    params: dict[str, Any] | None = None,
) -> PopulationConfig:
    """Build a single :class:`~arc3_agi.runner.PopulationConfig` for *pop_id*.

    The population's seed is ``population_seed + pop_id`` (or ``None`` when
    ``population_seed`` is ``None``), ensuring each population is fully
    reproducible yet independent.
    """
    params = _resolve_experiment_params(params)
    checkpoint_interval = int(params["checkpoint_interval"])
    ckpt_cfg = CheckpointConfig(
        enabled=checkpoint_interval > 0,
        generation_interval=checkpoint_interval,
    )
    return PopulationConfig(
        size=int(params["population_size"]),
        AutomatonClass=MazeAutomaton,
        environment=maze,
        ticks_per_restart=int(params["ticks_per_restart"]),
        restarts_per_gen=int(params["restarts_per_gen"]),
        checkpoint_config=ckpt_cfg,
        fingerprint_config=_fingerprint_config(params),
        automaton_params=dict(params.get("automaton_params", {})),
        seed=_population_seed(pop_id, params),
    )


def build_configs(
    maze: Maze,
    params: dict[str, Any] | None = None,
    *,
    count: int | None = None,
) -> list[PopulationConfig]:
    """Build :data:`MAX_PARALLEL` identical :class:`~arc3_agi.runner.PopulationConfig` objects.

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
        ``MAX_PARALLEL`` configs ready to pass to :func:`~arc3_agi.runner.launch_populations`.
    """
    params = _resolve_experiment_params(params)
    n = int(params["max_parallel"]) if count is None else count
    return [_build_config(i, maze, params) for i in range(n)]


def run(
    base_dir: Path = BASE_DIR,
    params: dict[str, Any] | None = None,
) -> list[PopulationHandle]:
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
    params = _resolve_experiment_params(params)
    max_parallel = int(params["max_parallel"])
    max_generations = int(params["max_generations"])
    ticks_per_restart = int(params["ticks_per_restart"])
    restarts_per_gen = int(params["restarts_per_gen"])
    checkpoint_interval = int(params["checkpoint_interval"])
    poll_interval_s = float(params["poll_interval_s"])
    tty = _is_tty()

    maze = Maze(
        name="MazeRunnerMaze",
        side_length_bits=int(params["side_length_bits"]),
        seed=params["maze_seed"],
    )
    configs = build_configs(maze, params, count=max_parallel)

    print(
        f"\nMaze Runner — {max_parallel} populations × {max_generations} generations "
        f"× {ticks_per_restart} ticks/restart × {restarts_per_gen} restart(s)/gen\n"
        f"  Maze: {maze.width}×{maze.height}  "
        f"Population size: {int(params['population_size'])}  "
        f"Checkpoint every: {checkpoint_interval} gens\n"
        f"  Checkpoints → {base_dir.resolve()}\n"
    )

    handles = launch_populations(
        configs, max_generations=max_generations, base_dir=base_dir
    )
    t0 = time.monotonic()

    prev_lines = _print_progress(
        handles, max_generations, elapsed_s=0.0, first=True, tty=tty
    )

    try:
        while any(h.is_running for h in handles):
            time.sleep(poll_interval_s)
            prev_lines = _print_progress(
                handles,
                max_generations,
                elapsed_s=time.monotonic() - t0,
                first=False,
                tty=tty,
                prev_lines=prev_lines,
            )
    except BaseException:
        stop_all(handles)
        raise

    # Final update after all processes have exited.
    _print_progress(
        handles,
        max_generations,
        elapsed_s=time.monotonic() - t0,
        first=False,
        tty=tty,
        prev_lines=prev_lines,
    )

    total_s = time.monotonic() - t0
    print(f"\nAll {max_parallel} populations finished in {total_s:.1f}s.")
    return handles


def run_pool(
    base_dir: Path = BASE_DIR,
    *,
    run_id: str | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[list[dict], str, Path]:
    """Launch :data:`TOTAL_POPULATIONS` populations with a concurrency cap.

    Runs :data:`TOTAL_POPULATIONS` independent populations in total, but keeps
    at most :data:`MAX_PARALLEL` running at any one time.  Whenever a running
    population finishes it is immediately replaced by a new one until the total
    quota has been met.

    All populations in a single invocation share one checkpoint directory so
    that their checkpoints are grouped under a common run identifier::

        <base_dir>/<run_id>/pop_0/
        <base_dir>/<run_id>/pop_1/
        ...
        <base_dir>/<run_id>/pop_{TOTAL_POPULATIONS-1}/

    Parameters
    ----------
    base_dir:
        Root directory for checkpoint output.  Defaults to :data:`BASE_DIR`.
    run_id:
        Optional pre-supplied run identifier.  When ``None`` (the default) a
        unique id is generated from the current timestamp and random hex.

    Returns
    -------
    tuple[list[dict], str, Path]
        ``(snapshots, run_id, run_dir)`` where *snapshots* is the final
        progress snapshot for every completed population (in completion order),
        *run_id* is the identifier used for this run, and *run_dir* is the
        absolute path to the run's checkpoint directory.
    """
    import secrets as _secrets
    from datetime import datetime as _datetime

    params = _resolve_experiment_params(params)
    total_populations = int(params["total_populations"])
    max_parallel = int(params["max_parallel"])
    max_generations = int(params["max_generations"])
    ticks_per_restart = int(params["ticks_per_restart"])
    restarts_per_gen = int(params["restarts_per_gen"])
    population_size = int(params["population_size"])
    checkpoint_interval = int(params["checkpoint_interval"])
    poll_interval_s = float(params["poll_interval_s"])
    tty = _is_tty()

    maze = Maze(
        name="MazeRunnerMaze",
        side_length_bits=int(params["side_length_bits"]),
        seed=params["maze_seed"],
    )

    # One shared run_id groups all checkpoint directories under a single folder.
    if run_id is None:
        run_id = _datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + _secrets.token_hex(3)

    print(
        f"\nMaze Runner (pool) — {total_populations} total × {max_parallel} parallel "
        f"× {max_generations} generations "
        f"× {ticks_per_restart} ticks/restart × {restarts_per_gen} restart(s)/gen\n"
        f"  Maze: {maze.width}×{maze.height}  "
        f"Population size: {population_size}  "
        f"Checkpoint every: {checkpoint_interval} gens\n"
        f"  Run ID: {run_id}   Checkpoints → {base_dir.resolve()}\n"
    )

    active: list[PopulationHandle] = []
    next_id: int = 0
    completed_snapshots: list[dict] = []

    # Initial fill — launch min(max_parallel, total_populations) populations.
    initial = min(max_parallel, total_populations)
    for _ in range(initial):
        config = _build_config(next_id, maze, params)
        [handle] = launch_populations(
            [config],
            max_generations=max_generations,
            base_dir=base_dir,
            run_id=run_id,
            start_pop_id=next_id,
        )
        active.append(handle)
        next_id += 1

    t0 = time.monotonic()
    prev_lines = _print_progress(
        active,
        max_generations,
        elapsed_s=0.0,
        first=True,
        tty=tty,
        completed_snapshots=completed_snapshots,
        total_populations=total_populations,
    )

    try:
        while active:
            time.sleep(poll_interval_s)

            still_running: list[PopulationHandle] = []
            for h in active:
                if h.is_running:
                    still_running.append(h)
                else:
                    # Drain the final progress snapshot before discarding the handle.
                    completed_snapshots.append(h.progress)
                    # Backfill the freed slot if the quota is not yet met.
                    if next_id < total_populations:
                        config = _build_config(next_id, maze, params)
                        [new_handle] = launch_populations(
                            [config],
                            max_generations=max_generations,
                            base_dir=base_dir,
                            run_id=run_id,
                            start_pop_id=next_id,
                        )
                        still_running.append(new_handle)
                        next_id += 1

            active = still_running
            prev_lines = _print_progress(
                active,
                max_generations,
                elapsed_s=time.monotonic() - t0,
                first=False,
                tty=tty,
                prev_lines=prev_lines,
                completed_snapshots=completed_snapshots,
                total_populations=total_populations,
            )
    except BaseException:
        stop_all(active)
        raise

    # Final progress update after the last population finishes.
    _print_progress(
        active,
        max_generations,
        elapsed_s=time.monotonic() - t0,
        first=False,
        tty=tty,
        prev_lines=prev_lines,
        completed_snapshots=completed_snapshots,
        total_populations=total_populations,
    )

    total_s = time.monotonic() - t0
    avg_s = total_s / total_populations if total_populations else 0.0
    print(
        f"\nAll {total_populations} populations finished in {total_s:.1f}s "
        f"({avg_s:.1f}s avg per population)."
    )
    run_dir = base_dir / run_id
    return completed_snapshots, run_id, run_dir


# ---------------------------------------------------------------------------
# Experiment entry point
# ---------------------------------------------------------------------------


DEFAULT_DB_PATH: Path = Path("experiments") / "runs.duckdb"
"""Default path for the experiment DuckDB database."""


def run_experiment(
    name: str,
    params: dict[str, Any],
    description: str = "",
    base_dir: Path = BASE_DIR,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    """Run a full pool experiment, persist results, and return the experiment id.

    Calls :func:`run_pool` and then ingests all per-population
    ``fitness_history.json`` files into the experiment database so the run
    can be queried and plotted in the analysis notebook.

    Parameters
    ----------
    name:
        Short human-readable experiment name, e.g. ``"baseline"``.
    params:
        Dictionary of experiment parameters to store in the database.
    description:
        Free-text description of the experiment's purpose and parameters.
    base_dir:
        Root directory for checkpoint output.  Defaults to :data:`BASE_DIR`.
    db_path:
        Path to the DuckDB experiment database.  Created automatically if it
        does not exist.  Defaults to :data:`DEFAULT_DB_PATH`.

    Returns
    -------
    int
        The experiment id assigned in the database, or the existing id when
        an experiment with the same name has already been recorded.
    """
    with ExperimentStore(db_path) as store:
        existing_id = store.get_experiment_id_by_name(name)
        if existing_id is not None:
            print(
                f"\nExperiment '{name}' already exists → id={existing_id}; "
                "skipping run."
            )
            return existing_id

    params = _resolve_experiment_params(params)
    snapshots, run_id, run_dir = run_pool(base_dir=base_dir, params=params)

    with ExperimentStore(db_path) as store:
        experiment_id = store.create_experiment(
            name=name,
            description=description,
            run_id=run_id,
            params=params,
        )
        n_rows = store.ingest_run(experiment_id, run_dir)

    print(
        f"\nExperiment '{name}' saved → id={experiment_id}  "
        f"({n_rows} generation-stat rows across {len(snapshots)} populations)"
        f"\n  DB: {Path(db_path).resolve()}"
    )
    return experiment_id


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    params = default_experiment_params()
    run_experiment(
        name="baseline-state4",
        params=params,
        description="Baseline maze evolution run.",
    )
    params["automaton_params"]["state_bits"] = 3
    run_experiment(
        name="baseline-state3",
        params=params,
        description="Baseline maze evolution run with 3 state bits.",
    )
    params["automaton_params"]["state_bits"] = 5
    run_experiment(
        name="baseline-state5",
        params=params,
        description="Baseline maze evolution run with 5 state bits.",
    )
