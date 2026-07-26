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

import time
from pathlib import Path
from typing import Any

from arc3_agi.batch_runner import generate_run_id as _generate_run_id
from arc3_agi.batch_runner import run_batch as _run_batch
from arc3_agi.batch_runner import run_pool as _run_population_pool
from arc3_agi.batch_runner import run_tracked_experiment as _run_tracked_experiment
from arc3_agi.checkpoint import CheckpointConfig
from arc3_agi.experiment import ExperimentStore
from arc3_agi.fingerprint import FingerprintConfig
from arc3_agi.maze import Maze, MazeAutomaton
from arc3_agi.runner import (
    PopulationConfig,
    PopulationHandle,
    launch_populations,
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

    t0 = time.monotonic()
    handles = _run_batch(
        configs,
        max_generations=max_generations,
        base_dir=base_dir,
        poll_interval_s=poll_interval_s,
        launch=launch_populations,
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
    params = _resolve_experiment_params(params)
    total_populations = int(params["total_populations"])
    max_parallel = int(params["max_parallel"])
    max_generations = int(params["max_generations"])
    ticks_per_restart = int(params["ticks_per_restart"])
    restarts_per_gen = int(params["restarts_per_gen"])
    population_size = int(params["population_size"])
    checkpoint_interval = int(params["checkpoint_interval"])
    poll_interval_s = float(params["poll_interval_s"])

    maze = Maze(
        name="MazeRunnerMaze",
        side_length_bits=int(params["side_length_bits"]),
        seed=params["maze_seed"],
    )

    # One shared run_id groups all checkpoint directories under a single folder.
    if run_id is None:
        run_id = _generate_run_id()

    print(
        f"\nMaze Runner (pool) — {total_populations} total × {max_parallel} parallel "
        f"× {max_generations} generations "
        f"× {ticks_per_restart} ticks/restart × {restarts_per_gen} restart(s)/gen\n"
        f"  Maze: {maze.width}×{maze.height}  "
        f"Population size: {population_size}  "
        f"Checkpoint every: {checkpoint_interval} gens\n"
        f"  Run ID: {run_id}   Checkpoints → {base_dir.resolve()}\n"
    )

    t0 = time.monotonic()
    completed_snapshots, run_id, run_dir = _run_population_pool(
        total_populations=total_populations,
        max_parallel=max_parallel,
        max_generations=max_generations,
        base_dir=base_dir,
        config_factory=lambda pop_id: _build_config(pop_id, maze, params),
        poll_interval_s=poll_interval_s,
        run_id=run_id,
        launch=launch_populations,
    )

    total_s = time.monotonic() - t0
    avg_s = total_s / total_populations if total_populations else 0.0
    print(
        f"\nAll {total_populations} populations finished in {total_s:.1f}s "
        f"({avg_s:.1f}s avg per population)."
    )
    return completed_snapshots, run_id, run_dir


# ---------------------------------------------------------------------------
# Experiment entry point
# ---------------------------------------------------------------------------


def run_experiment(
    name: str,
    params: dict[str, Any],
    description: str = "",
    base_dir: Path = BASE_DIR,
    database_url: str | None = None,
) -> int:
    """Run a full pool experiment, persist results, and return the experiment id.

    Calls :func:`run_pool` and then ingests all per-population
    ``fitness_history.json`` files into PostgreSQL so the run can be queried
    and plotted in the analysis notebook.  The experiment name is claimed in
    the database before any population subprocess starts.

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
    database_url:
        PostgreSQL connection URL.  When omitted, :envvar:`DATABASE_URL` is
        used, falling back to the local development default.

    Returns
    -------
    int
        The experiment id assigned in the database, or the existing id when
        a completed experiment with the same name has already been recorded.
    """
    params = _resolve_experiment_params(params)
    return _run_tracked_experiment(
        name=name,
        params=params,
        description=description,
        base_dir=base_dir,
        database_url=database_url,
        pool_runner=run_pool,
        store_factory=ExperimentStore,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    params = default_experiment_params()
    params["fingerprint_enabled"] = True
    params["max_generations"] = 1000
    run_experiment(
        name="baseline-state4-tsetlin",
        params=params,
        description="Baseline maze evolution run.",
    )
    params["automaton_params"]["state_bits"] = 3
    run_experiment(
        name="baseline-state3-tsetlin",
        params=params,
        description="Baseline maze evolution run with 3 state bits.",
    )
    params["automaton_params"]["state_bits"] = 5
    run_experiment(
        name="baseline-state5-tsetlin",
        params=params,
        description="Baseline maze evolution run with 5 state bits.",
    )
