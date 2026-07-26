"""Parallel batch and experiment runner for WikiText populations."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from arc3_agi.batch_runner import generate_run_id as _generate_run_id
from arc3_agi.batch_runner import run_batch as _run_batch
from arc3_agi.batch_runner import run_pool as _run_population_pool
from arc3_agi.batch_runner import run_tracked_experiment as _run_tracked_experiment
from arc3_agi.checkpoint import CheckpointConfig
from arc3_agi.environment import ByteEnv
from arc3_agi.experiment import ExperimentStore
from arc3_agi.fingerprint import FingerprintConfig
from arc3_agi.runner import PopulationConfig, PopulationHandle, launch_populations
from arc3_agi.wiki_text_2 import (
    WIKITEXT_DATASET_CONFIG,
    WIKITEXT_DATASET_NAME,
    WIKITEXT_SPLIT,
    WikiAutomaton,
    load_wikitext_environment,
)

MAX_PARALLEL: int = 12
"""Maximum number of populations to evolve concurrently."""

TOTAL_POPULATIONS: int = 100
"""Total number of populations to run in pool mode."""

MAX_GENERATIONS: int = 1000
"""Number of evaluate/evolve cycles for each population."""

TICKS_PER_RESTART: int = 1000
"""Number of WikiText byte predictions made during each restart."""

RESTARTS_PER_GEN: int = 1
"""Number of independently reset evaluations averaged per generation."""

POPULATION_SIZE: int = 100
"""Number of WikiText predictors in each population."""

POPULATION_SEED: int | None = 0
"""Base deterministic seed; population i receives this value plus i."""

FINGERPRINT_ENABLED: bool = True
"""Whether to enable fingerprint-guided mate selection."""

FINGERPRINT_BITS: int = 4
"""Bit width of the mate-selection fingerprint."""

FINGERPRINT_TOURNAMENT_K: int = 4
"""Tournament size for fingerprint-guided mate selection."""

FINGERPRINT_MUTATION_RATE: float = 0.01
"""Per-bit mutation probability for inherited selection fingerprints."""

AUTOMATON_PARAMS: dict[str, Any] = {
    "env_bits": 16,
    "state_bits": 8,
    "resp_bits": 8,
    "num_clauses": 16,
}
"""Keyword arguments forwarded to each Wiki automaton."""

CHECKPOINT_INTERVAL: int = MAX_GENERATIONS
"""Write a checkpoint every this many generations; zero disables them."""

POLL_INTERVAL_S: float = 2.0
"""Seconds between progress-table refreshes."""

BASE_DIR: Path = Path("runs")
"""Root directory for per-run artifacts."""


def default_experiment_params() -> dict[str, Any]:
    """Return the default WikiText experiment parameters."""
    return {
        "total_populations": TOTAL_POPULATIONS,
        "max_parallel": MAX_PARALLEL,
        "max_generations": MAX_GENERATIONS,
        "ticks_per_restart": TICKS_PER_RESTART,
        "restarts_per_gen": RESTARTS_PER_GEN,
        "population_size": POPULATION_SIZE,
        "population_seed": POPULATION_SEED,
        "fingerprint_enabled": FINGERPRINT_ENABLED,
        "fingerprint_bits": FINGERPRINT_BITS,
        "fingerprint_tournament_k": FINGERPRINT_TOURNAMENT_K,
        "fingerprint_mutation_rate": FINGERPRINT_MUTATION_RATE,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "poll_interval_s": POLL_INTERVAL_S,
        "dataset_name": WIKITEXT_DATASET_NAME,
        "dataset_config": WIKITEXT_DATASET_CONFIG,
        "dataset_split": WIKITEXT_SPLIT,
        "automaton_params": dict(AUTOMATON_PARAMS),
    }


def _resolve_experiment_params(params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge caller-supplied values over the WikiText defaults."""
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


def _population_seed(population_id: int, params: dict[str, Any]) -> int | None:
    seed = params.get("population_seed")
    return int(seed) + population_id if seed is not None else None


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


def _load_environment(params: dict[str, Any]) -> ByteEnv:
    return load_wikitext_environment(
        str(params["dataset_name"]),
        str(params["dataset_config"]),
        str(params["dataset_split"]),
    )


def _build_config(
    population_id: int,
    environment: ByteEnv,
    params: dict[str, Any] | None = None,
) -> PopulationConfig:
    """Build one reproducibly seeded WikiText population configuration."""
    params = _resolve_experiment_params(params)
    checkpoint_interval = int(params["checkpoint_interval"])
    checkpoint_config = CheckpointConfig(
        enabled=checkpoint_interval > 0,
        generation_interval=checkpoint_interval,
    )
    return PopulationConfig(
        size=int(params["population_size"]),
        AutomatonClass=WikiAutomaton,
        environment=environment,
        ticks_per_restart=int(params["ticks_per_restart"]),
        restarts_per_gen=int(params["restarts_per_gen"]),
        checkpoint_config=checkpoint_config,
        fingerprint_config=_fingerprint_config(params),
        automaton_params=dict(params.get("automaton_params", {})),
        seed=_population_seed(population_id, params),
    )


def build_configs(
    environment: ByteEnv,
    params: dict[str, Any] | None = None,
    *,
    count: int | None = None,
) -> list[PopulationConfig]:
    """Build WikiText configs that share one immutable byte environment."""
    params = _resolve_experiment_params(params)
    config_count = int(params["max_parallel"]) if count is None else count
    return [
        _build_config(population_id, environment, params)
        for population_id in range(config_count)
    ]


def run(
    base_dir: Path = BASE_DIR,
    params: dict[str, Any] | None = None,
) -> list[PopulationHandle]:
    """Launch one fixed parallel batch and wait for all populations."""
    params = _resolve_experiment_params(params)
    max_parallel = int(params["max_parallel"])
    max_generations = int(params["max_generations"])
    ticks_per_restart = int(params["ticks_per_restart"])
    restarts_per_gen = int(params["restarts_per_gen"])
    checkpoint_interval = int(params["checkpoint_interval"])
    poll_interval_s = float(params["poll_interval_s"])
    environment = _load_environment(params)
    configs = build_configs(environment, params, count=max_parallel)

    print(
        f"\nWikiText Runner — {max_parallel} populations × "
        f"{max_generations} generations × {ticks_per_restart} ticks/restart × "
        f"{restarts_per_gen} restart(s)/gen\n"
        f"  Dataset: {params['dataset_name']}/{params['dataset_config']} "
        f"[{params['dataset_split']}]  Population size: "
        f"{int(params['population_size'])}\n"
        f"  Checkpoint every: {checkpoint_interval} gens  "
        f"Checkpoints → {base_dir.resolve()}\n"
    )

    started = time.monotonic()
    handles = _run_batch(
        configs,
        max_generations=max_generations,
        base_dir=base_dir,
        poll_interval_s=poll_interval_s,
        launch=launch_populations,
    )
    elapsed_s = time.monotonic() - started
    print(f"\nAll {max_parallel} populations finished in {elapsed_s:.1f}s.")
    return handles


def run_pool(
    base_dir: Path = BASE_DIR,
    *,
    run_id: str | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str, Path]:
    """Run a bounded pool of WikiText populations to completion."""
    params = _resolve_experiment_params(params)
    total_populations = int(params["total_populations"])
    max_parallel = int(params["max_parallel"])
    max_generations = int(params["max_generations"])
    ticks_per_restart = int(params["ticks_per_restart"])
    restarts_per_gen = int(params["restarts_per_gen"])
    checkpoint_interval = int(params["checkpoint_interval"])
    poll_interval_s = float(params["poll_interval_s"])
    environment = _load_environment(params)
    if run_id is None:
        run_id = _generate_run_id()

    print(
        f"\nWikiText Runner (pool) — {total_populations} total × "
        f"{max_parallel} parallel × {max_generations} generations × "
        f"{ticks_per_restart} ticks/restart × {restarts_per_gen} restart(s)/gen\n"
        f"  Dataset: {params['dataset_name']}/{params['dataset_config']} "
        f"[{params['dataset_split']}]  Population size: "
        f"{int(params['population_size'])}\n"
        f"  Checkpoint every: {checkpoint_interval} gens  Run ID: {run_id}\n"
        f"  Checkpoints → {base_dir.resolve()}\n"
    )

    started = time.monotonic()
    snapshots, run_id, run_dir = _run_population_pool(
        total_populations=total_populations,
        max_parallel=max_parallel,
        max_generations=max_generations,
        base_dir=base_dir,
        config_factory=lambda population_id: _build_config(
            population_id, environment, params
        ),
        poll_interval_s=poll_interval_s,
        run_id=run_id,
        launch=launch_populations,
    )
    elapsed_s = time.monotonic() - started
    average_s = elapsed_s / total_populations if total_populations else 0.0
    print(
        f"\nAll {total_populations} populations finished in {elapsed_s:.1f}s "
        f"({average_s:.1f}s avg per population)."
    )
    return snapshots, run_id, run_dir


def run_experiment(
    name: str,
    params: dict[str, Any],
    description: str = "",
    base_dir: Path = BASE_DIR,
    database_url: str | None = None,
) -> int:
    """Run, persist, and return the id of a tracked WikiText experiment."""
    resolved_params = _resolve_experiment_params(params)
    return _run_tracked_experiment(
        name=name,
        params=resolved_params,
        description=description,
        base_dir=base_dir,
        database_url=database_url,
        pool_runner=run_pool,
        store_factory=ExperimentStore,
    )


def main() -> None:
    """Run the default tracked WikiText baseline."""
    run_experiment(
        name="wikitext2-baseline-state8-tsetlin",
        params=default_experiment_params(),
        description="Baseline WikiText 2 byte-prediction evolution run.",
    )


if __name__ == "__main__":
    main()
