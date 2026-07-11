"""Parallel background population runner.

Provides :class:`PopulationConfig`, :class:`PopulationHandle`, and
:func:`launch_populations` / :func:`wait_all` for running multiple independent
populations concurrently in separate OS processes.

Each launched population runs entirely in the background — the caller gets back
a list of :class:`PopulationHandle` objects immediately and can poll their
progress or block on completion at any time.

Usage example::

    from pathlib import Path
    from arc3_agi.runner import PopulationConfig, launch_populations, wait_all
    from arc3_agi.maze import Maze, MazeAutomaton
    from arc3_agi.checkpoint import CheckpointConfig
    from arc3_agi.fingerprint import FingerprintConfig

    maze = Maze(name="example", side_length_bits=4, seed=42)
    config = PopulationConfig(
        size=100,
        AutomatonClass=MazeAutomaton,
        environment=maze,
        ticks_per_restart=100,
        restarts_per_gen=3,
        checkpoint_config=CheckpointConfig(generation_interval=10),
        fingerprint_config=FingerprintConfig(bits=4, tournament_k=4),
    )

    handles = launch_populations([config, config], max_generations=500)

    # Poll while doing other work.
    for h in handles:
        print(h.population_id, h.progress)

    # Block until all populations have finished.
    wait_all(handles)
"""

from __future__ import annotations

import math
import multiprocessing
import random
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty
from typing import Any

from arc3_agi.automaton import AutomatonBase
from arc3_agi.checkpoint import CheckpointConfig
from arc3_agi.environment import Environment
from arc3_agi.fingerprint import FingerprintConfig
from arc3_agi.population import Population

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class PopulationConfig:
    """All parameters needed to instantiate and run a :class:`Population`.

    All fields must be picklable so that they can be transmitted to the
    worker subprocess on platforms that use the ``spawn`` or ``forkserver``
    start method.

    Attributes:
        size:             Number of automata in the population.
        AutomatonClass:   Concrete :class:`~arc3_agi.automaton.AutomatonBase`
                          subclass to instantiate for each member.
        environment:      Environment instance shared by all automata in this
                          population.
        ticks_per_restart: Number of :meth:`~Population.tick` calls executed
                          within each restart.
        restarts_per_gen: Number of independent restarts per generation.  The
                          fitness used by :meth:`~Population.evolve` is the
                          mean across all restarts.  Defaults to 1 (original
                          single-attempt behaviour).
        checkpoint_config: Optional checkpoint settings.  The runner overrides
                          ``base_dir`` to an isolated per-population
                          subdirectory; ``enabled`` and ``generation_interval``
                          are preserved as supplied.
        fingerprint_config: Optional fingerprint / mate-selection settings.
    """

    size: int
    AutomatonClass: type[AutomatonBase]
    environment: Environment
    ticks_per_restart: int
    restarts_per_gen: int = 1
    checkpoint_config: CheckpointConfig | None = None
    fingerprint_config: FingerprintConfig | None = None
    seed: int | None = None
    """Optional integer seed for fully deterministic, reproducible evolution.

    When set, every source of randomness in the population subprocess —
    mate selection, automaton RNGs, genetic-code crossover, fingerprints,
    and starting positions — is derived from this seed.  Two runs with the
    same seed and the same config will produce byte-identical results.
    """


# ---------------------------------------------------------------------------
# Worker function — must be at module level for multiprocessing pickling
# ---------------------------------------------------------------------------


def _worker_fn(
    config: PopulationConfig,
    max_generations: int,
    pop_dir: Path,
    pop_id: int,
    queue: "multiprocessing.Queue[dict[str, Any]]",
    seed: int | None = None,
) -> None:
    """Entry point for each population subprocess.

    Creates a :class:`Population` from *config*, runs it for *max_generations*
    tick/evolve cycles, and pushes a progress dict to *queue* after every
    generation.  The final push always carries ``"is_running": False``.

    On an unhandled exception the worker pushes an error dict (also with
    ``"is_running": False``) before re-raising so the caller can detect the
    failure via :attr:`PopulationHandle.progress`.

    Parameters
    ----------
    config:
        Full population configuration.
    max_generations:
        Number of complete tick/evolve cycles to run.
    pop_dir:
        Isolated directory for this population's checkpoints.  The
        :class:`~arc3_agi.population.Population` will create a
        timestamped subdirectory inside it.
    pop_id:
        Zero-based index of this population in the batch (informational).
    queue:
        Multiprocessing queue used to push progress updates to the parent.
    """
    # Seed the global Python random module so any module-level random calls
    # (including those not yet migrated to instance RNGs) are deterministic.
    if seed is not None:
        random.seed(seed)

    # Build an isolated checkpoint config that writes under pop_dir.
    user_ckpt = config.checkpoint_config
    if user_ckpt is None:
        ckpt = CheckpointConfig(base_dir=pop_dir)
    else:
        ckpt = CheckpointConfig(
            enabled=user_ckpt.enabled,
            base_dir=pop_dir,
            generation_interval=user_ckpt.generation_interval,
        )

    try:
        population = Population(
            size=config.size,
            AutomatonClass=config.AutomatonClass,
            environment=config.environment,
            checkpoint_config=ckpt,
            fingerprint_config=config.fingerprint_config,
            seed=seed,
        )

        for gen in range(max_generations):
            population.run_generation(config.ticks_per_restart, config.restarts_per_gen)
            fitnesses = population.evolve()
            n = len(fitnesses)
            queue.put(
                {
                    "generation": gen + 1,
                    "min_fitness": min(fitnesses),
                    "max_fitness": max(fitnesses),
                    "mean_fitness": sum(fitnesses) / n if n else 0.0,
                    "is_running": gen + 1 < max_generations,
                }
            )

    except Exception as exc:  # noqa: BLE001
        queue.put({"error": str(exc), "is_running": False})
        raise


# ---------------------------------------------------------------------------
# Handle returned to the caller per launched population
# ---------------------------------------------------------------------------


class PopulationHandle:
    """A handle to a single background population process.

    Returned by :func:`launch_populations`.  The handle is the sole public
    interface between the caller and the subprocess — no shared memory or
    locks are exposed.

    Attributes:
        population_id: Zero-based index of this population in the batch.
    """

    def __init__(
        self,
        population_id: int,
        process: "multiprocessing.Process",
        queue: "multiprocessing.Queue[dict[str, Any]]",
    ) -> None:
        self.population_id = population_id
        self._process = process
        self._queue = queue
        # Seed the cache so callers always get a complete dict on first access.
        self._latest: dict[str, Any] = {
            "generation": 0,
            "min_fitness": float("nan"),
            "max_fitness": float("nan"),
            "mean_fitness": float("nan"),
            "best_max_fitness": float("nan"),
            "best_mean_fitness": float("nan"),
            "is_running": True,
        }

    @property
    def progress(self) -> dict[str, Any]:
        """Return the latest progress snapshot for this population.

        Drains all pending items from the internal queue (non-blocking) and
        keeps only the most recent one.  The returned dict always contains:

        * ``"generation"`` (*int*) — number of completed generations so far.
        * ``"min_fitness"`` / ``"max_fitness"`` / ``"mean_fitness"`` (*float*).
        * ``"is_running"`` (*bool*) — ``False`` once the worker has finished
          all generations (or encountered an error).
        * ``"error"`` (*str*, optional) — only present if the worker raised.
        """
        best_max = self._latest.get("best_max_fitness", float("nan"))
        best_mean = self._latest.get("best_mean_fitness", float("nan"))
        latest: dict[str, Any] | None = None
        try:
            while True:
                item = self._queue.get_nowait()
                item_max = item.get("max_fitness", float("nan"))
                item_mean = item.get("mean_fitness", float("nan"))
                if not math.isnan(item_max) and (
                    math.isnan(best_max) or item_max > best_max
                ):
                    best_max = item_max
                if not math.isnan(item_mean) and (
                    math.isnan(best_mean) or item_mean > best_mean
                ):
                    best_mean = item_mean
                latest = item
        except Empty:
            pass
        if latest is not None:
            latest["best_max_fitness"] = best_max
            latest["best_mean_fitness"] = best_mean
            self._latest = latest
        return dict(self._latest)

    @property
    def is_running(self) -> bool:
        """``True`` while the subprocess is still alive."""
        return self._process.is_alive()

    def wait(self, timeout: float | None = None) -> None:
        """Block until this population's subprocess exits.

        Parameters
        ----------
        timeout:
            Maximum seconds to wait.  ``None`` (the default) means wait
            indefinitely.
        """
        self._process.join(timeout)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def launch_populations(
    configs: list[PopulationConfig],
    max_generations: int,
    base_dir: Path = Path("runs"),
) -> list[PopulationHandle]:
    """Spawn one background OS process per config and return handles immediately.

    Populations are fully isolated: each runs in its own subprocess with its
    own memory space.  Checkpoints are written to::

        <base_dir>/<run_id>/pop_<i>/

    where ``run_id`` is a unique identifier for this batch (ISO timestamp +
    random hex suffix to avoid collisions when multiple batches start within
    the same second).

    Parameters
    ----------
    configs:
        One :class:`PopulationConfig` per population to launch.  Configs may
        differ freely (different automaton class, environment, size, etc.).
    max_generations:
        Number of tick/evolve cycles each population runs before stopping.
    base_dir:
        Root directory under which per-batch run folders are created.
        Created automatically if it does not exist.

    Returns
    -------
    list[PopulationHandle]
        One handle per config, in the same order as *configs*.
    """
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + secrets.token_hex(3)
    handles: list[PopulationHandle] = []

    for i, config in enumerate(configs):
        pop_dir = base_dir / run_id / f"pop_{i}"
        q: multiprocessing.Queue[dict[str, Any]] = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=_worker_fn,
            args=(config, max_generations, pop_dir, i, q, config.seed),
            daemon=True,
            name=f"pop-{run_id}-{i}",
        )
        p.start()
        handles.append(PopulationHandle(i, p, q))

    return handles


def wait_all(
    handles: list[PopulationHandle],
    timeout: float | None = None,
) -> None:
    """Block until every population in *handles* has finished.

    Parameters
    ----------
    handles:
        List of :class:`PopulationHandle` objects returned by
        :func:`launch_populations`.
    timeout:
        Per-handle join timeout in seconds.  ``None`` (the default) means
        wait indefinitely for each handle in sequence.
    """
    for handle in handles:
        handle.wait(timeout)
