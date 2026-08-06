"""Shared orchestration for parallel population experiments."""

from __future__ import annotations

import math
import secrets
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from arc3_agi.experiment import ExperimentStore
from arc3_agi.runner import (
    PopulationConfig,
    PopulationHandle,
    launch_populations,
    stop_all,
)

_CSI = "\033["
_CURSOR_UP = _CSI + "{}A"
_ERASE_LINE = _CSI + "2K\r"

LaunchPopulations = Callable[..., list[PopulationHandle]]
ConfigFactory = Callable[[int], PopulationConfig]
PoolRunner = Callable[..., tuple[list[dict[str, Any]], str, Path]]
StoreFactory = Callable[[str | None], Any]


def generate_run_id() -> str:
    """Return a unique identifier suitable for a run directory."""
    return datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + secrets.token_hex(3)


def _is_tty() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def format_progress_table(
    handles: list[PopulationHandle],
    max_generations: int,
    elapsed_s: float,
    completed_snapshots: list[dict[str, Any]] | None = None,
    total_populations: int | None = None,
) -> list[str]:
    """Return the terminal status table for active and completed populations."""
    done = sum(1 for handle in handles if not handle.is_running)
    if completed_snapshots is None:
        header = (
            f"  Elapsed: {elapsed_s:6.0f}s   "
            f"Finished: {done}/{len(handles)} populations"
        )
    else:
        total = total_populations if total_populations is not None else len(handles)
        header = (
            f"  Elapsed: {elapsed_s:6.0f}s   Active: {done}/{len(handles)} done"
            f"   Completed: {len(completed_snapshots)}/{total} total"
        )
    lines = [
        header,
        f"  {'Pop':>3}  {'Gen':>6}/{max_generations:<6}  {'Max fit':>9}  "
        f"{'Mean fit':>9}  {'Best Max':>9}  {'Best Mean':>9}  {'Status':<8}",
        "  " + "-" * 78,
    ]

    sum_max = sum_mean = sum_best_max = sum_best_mean = 0.0
    n_valid = 0
    snapshots = [handle.progress for handle in handles]

    for handle, progress in zip(handles, snapshots):
        generation = progress.get("generation", 0)
        max_fitness = progress.get("max_fitness", float("nan"))
        mean_fitness = progress.get("mean_fitness", float("nan"))
        best_max = progress.get("best_max_fitness", float("nan"))
        best_mean = progress.get("best_mean_fitness", float("nan"))
        status = "done" if not handle.is_running else "running"
        percent = 100 * generation / max_generations if max_generations else 0
        bar_filled = int(percent / 10)
        bar = "[" + "#" * bar_filled + "." * (10 - bar_filled) + "]"
        lines.append(
            f"  {handle.population_id:>3}  {generation:>6}/{max_generations:<6}  "
            f"{max_fitness:>9.3f}  {mean_fitness:>9.3f}  {best_max:>9.3f}  "
            f"{best_mean:>9.3f}  {status:<8}  {percent:5.1f}% {bar}"
        )
        values = (max_fitness, mean_fitness, best_max, best_mean)
        if all(not math.isnan(value) for value in values):
            sum_max += max_fitness
            sum_mean += mean_fitness
            sum_best_max += best_max
            sum_best_mean += best_mean
            n_valid += 1

    lines.append("  " + "-" * 78)
    total = total_populations if total_populations is not None else len(handles)
    if completed_snapshots:
        completed_best_max = [
            snapshot.get("best_max_fitness", float("nan"))
            for snapshot in completed_snapshots
        ]
        completed_best_mean = [
            snapshot.get("best_mean_fitness", float("nan"))
            for snapshot in completed_snapshots
        ]
        valid_best_max = [
            value for value in completed_best_max if not math.isnan(value)
        ]
        valid_best_mean = [
            value for value in completed_best_mean if not math.isnan(value)
        ]
        average_best_max = (
            sum(valid_best_max) / len(valid_best_max)
            if valid_best_max
            else float("nan")
        )
        average_best_mean = (
            sum(valid_best_mean) / len(valid_best_mean)
            if valid_best_mean
            else float("nan")
        )
        lines.append(
            f"  {'':>3}  Completed: {len(completed_snapshots)}/{total}"
            f"   avg best_max: {average_best_max:>9.3f}"
            f"   avg best_mean: {average_best_mean:>9.3f}"
        )
    if n_valid:
        lines.append(
            f"  {'AVG':>3}  {'':>6} {'':6}  "
            f"{sum_max / n_valid:>9.3f}  {sum_mean / n_valid:>9.3f}  "
            f"{sum_best_max / n_valid:>9.3f}  "
            f"{sum_best_mean / n_valid:>9.3f}  {'':8}"
        )
    else:
        lines.append(f"  {'AVG':>3}  (no data yet)")
    return lines


def print_progress(
    handles: list[PopulationHandle],
    max_generations: int,
    elapsed_s: float,
    *,
    first: bool = False,
    tty: bool,
    prev_lines: int = 0,
    completed_snapshots: list[dict[str, Any]] | None = None,
    total_populations: int | None = None,
) -> int:
    """Render a progress table and return the number of printed lines."""
    lines = format_progress_table(
        handles,
        max_generations,
        elapsed_s,
        completed_snapshots,
        total_populations,
    )
    if tty and not first and prev_lines:
        sys.stdout.write(_CURSOR_UP.format(prev_lines))
        for _ in range(prev_lines):
            sys.stdout.write(_ERASE_LINE + "\n")
        sys.stdout.write(_CURSOR_UP.format(prev_lines))
    for line in lines:
        sys.stdout.write(line + "\n")
    sys.stdout.flush()
    return len(lines)


def run_batch(
    configs: list[PopulationConfig],
    *,
    max_generations: int,
    base_dir: Path,
    poll_interval_s: float,
    launch: LaunchPopulations = launch_populations,
) -> list[PopulationHandle]:
    """Launch a fixed batch, report progress, and wait for completion."""
    handles = launch(configs, max_generations=max_generations, base_dir=base_dir)
    started = time.monotonic()
    tty = _is_tty()
    previous_lines = print_progress(
        handles, max_generations, elapsed_s=0.0, first=True, tty=tty
    )

    try:
        while any(handle.is_running for handle in handles):
            time.sleep(poll_interval_s)
            previous_lines = print_progress(
                handles,
                max_generations,
                elapsed_s=time.monotonic() - started,
                tty=tty,
                prev_lines=previous_lines,
            )
    except BaseException:
        stop_all(handles)
        raise

    print_progress(
        handles,
        max_generations,
        elapsed_s=time.monotonic() - started,
        tty=tty,
        prev_lines=previous_lines,
    )
    return handles


def run_pool(
    *,
    total_populations: int,
    max_parallel: int,
    max_generations: int,
    base_dir: Path,
    config_factory: ConfigFactory,
    poll_interval_s: float,
    run_id: str | None = None,
    launch: LaunchPopulations = launch_populations,
) -> tuple[list[dict[str, Any]], str, Path]:
    """Run a bounded population pool and return final progress snapshots."""
    if run_id is None:
        run_id = generate_run_id()

    active: list[PopulationHandle] = []
    completed_snapshots: list[dict[str, Any]] = []
    next_id = 0
    for _ in range(min(max_parallel, total_populations)):
        [handle] = launch(
            [config_factory(next_id)],
            max_generations=max_generations,
            base_dir=base_dir,
            run_id=run_id,
            start_pop_id=next_id,
        )
        active.append(handle)
        next_id += 1

    started = time.monotonic()
    tty = _is_tty()
    previous_lines = print_progress(
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
            for handle in active:
                if handle.is_running:
                    still_running.append(handle)
                    continue

                completed_snapshots.append(handle.progress)
                if next_id < total_populations:
                    [new_handle] = launch(
                        [config_factory(next_id)],
                        max_generations=max_generations,
                        base_dir=base_dir,
                        run_id=run_id,
                        start_pop_id=next_id,
                    )
                    still_running.append(new_handle)
                    next_id += 1

            active = still_running
            previous_lines = print_progress(
                active,
                max_generations,
                elapsed_s=time.monotonic() - started,
                tty=tty,
                prev_lines=previous_lines,
                completed_snapshots=completed_snapshots,
                total_populations=total_populations,
            )
    except BaseException:
        stop_all(active)
        raise

    print_progress(
        active,
        max_generations,
        elapsed_s=time.monotonic() - started,
        tty=tty,
        prev_lines=previous_lines,
        completed_snapshots=completed_snapshots,
        total_populations=total_populations,
    )
    return completed_snapshots, run_id, base_dir / run_id


def run_tracked_experiment(
    *,
    name: str,
    params: dict[str, Any],
    description: str,
    base_dir: Path,
    database_url: str | None,
    pool_runner: PoolRunner,
    store_factory: StoreFactory = ExperimentStore,
) -> int:
    """Claim, run, ingest, and finalize one tracked experiment."""
    run_id = generate_run_id()
    experiment_id: int | None = None

    try:
        with store_factory(database_url) as store:
            claim = store.claim_experiment(
                name=name,
                description=description,
                run_id=run_id,
                params=params,
            )
            experiment_id = claim.experiment_id
            if claim.already_completed:
                print(
                    f"\nExperiment '{name}' already completed → id={experiment_id}; "
                    "skipping run."
                )
                return experiment_id
            store.mark_experiment_running(experiment_id)

        snapshots, _, run_dir = pool_runner(
            base_dir=base_dir,
            run_id=run_id,
            params=params,
        )

        with store_factory(database_url) as store:
            row_count = store.ingest_run(experiment_id, run_dir)
            diagnostics_count = 0
            if hasattr(store, "ingest_checkpoint_diagnostics"):
                diagnostics_count = store.ingest_checkpoint_diagnostics(
                    experiment_id,
                    run_dir,
                )
            store.mark_experiment_completed(experiment_id)
    except BaseException as exc:
        if experiment_id is not None:
            try:
                with store_factory(database_url) as store:
                    store.mark_experiment_failed(
                        experiment_id, f"{type(exc).__name__}: {exc}"
                    )
            except Exception as mark_error:
                print(
                    f"\nFailed to mark experiment '{name}' failed: {mark_error}",
                    file=sys.stderr,
                )
        raise

    print(
        f"\nExperiment '{name}' saved → id={experiment_id}  "
        f"({row_count} generation-stat rows, {diagnostics_count} checkpoint diagnostics "
        f"across {len(snapshots)} populations)"
        "\n  DB: PostgreSQL"
    )
    return experiment_id
