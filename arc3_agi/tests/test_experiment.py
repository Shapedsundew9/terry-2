"""Tests for experiment database helpers and runner entry points."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest

from arc3_agi import maze_runner
from arc3_agi.experiment import ExperimentClaim, ExperimentClaimError, ExperimentStore


def _unique_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


@pytest.fixture()
def database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("set TEST_DATABASE_URL or DATABASE_URL to run PostgreSQL tests")
    return url


def test_get_experiment_id_by_name_returns_existing_id(database_url: str) -> None:
    name = _unique_name("baseline")
    with ExperimentStore(database_url) as store:
        existing_id = store.create_experiment(name=name, params={"seed": 1})
        try:
            assert store.get_experiment_id_by_name(name) == existing_id
            assert store.get_experiment_id_by_name(_unique_name("missing")) is None
        finally:
            store.delete_experiment(existing_id)


def test_claim_experiment_blocks_non_completed_duplicates(
    database_url: str,
) -> None:
    name = _unique_name("claim")
    with ExperimentStore(database_url) as first_store:
        claim = first_store.claim_experiment(name=name, params={"seed": 1})
        try:
            assert claim.status == "claimed"
            assert claim.already_completed is False

            with ExperimentStore(database_url) as second_store:
                with pytest.raises(ExperimentClaimError, match="already exists"):
                    second_store.claim_experiment(name=name, params={"seed": 2})
        finally:
            first_store.delete_experiment(claim.experiment_id)


def test_claim_experiment_returns_completed_existing_id(database_url: str) -> None:
    name = _unique_name("completed")
    with ExperimentStore(database_url) as store:
        claim = store.claim_experiment(name=name, params={"seed": 1})
        store.mark_experiment_completed(claim.experiment_id)
        try:
            completed = store.claim_experiment(name=name, params={"seed": 2})

            assert completed.experiment_id == claim.experiment_id
            assert completed.status == "completed"
            assert completed.already_completed is True
        finally:
            store.delete_experiment(claim.experiment_id)


def test_ingest_run_upserts_generation_stats(
    tmp_path: Path,
    database_url: str,
) -> None:
    name = _unique_name("ingest")
    run_dir = tmp_path / "run-1"
    pop_dir = run_dir / "pop_0"
    pop_dir.mkdir(parents=True)
    (pop_dir / "fitness_history.json").write_text(
        json.dumps(
            {
                "pop_id": 0,
                "history": [
                    {
                        "generation": 1,
                        "min_fitness": 0.1,
                        "max_fitness": 0.8,
                        "mean_fitness": 0.4,
                        "duration_s": 1.2,
                    }
                ],
            }
        )
    )

    with ExperimentStore(database_url) as store:
        experiment_id = store.create_experiment(name=name, run_id="run-1")
        try:
            assert store.ingest_run(experiment_id, run_dir) == 1
            assert store.ingest_run(experiment_id, run_dir) == 1
            stats = store.load_stats(experiment_id)

            assert len(stats) == 1
            assert stats.iloc[0]["generation"] == 1
            assert stats.iloc[0]["max_fitness"] == pytest.approx(0.8)
        finally:
            store.delete_experiment(experiment_id)


def test_run_experiment_skips_completed_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []

    class FakeExperimentStore:
        def __init__(self, _database_url: str | None = None) -> None:
            events.append("open")

        def __enter__(self) -> "FakeExperimentStore":
            return self

        def __exit__(self, *_args: Any) -> None:
            events.append("close")

        def claim_experiment(self, **_kwargs: Any) -> ExperimentClaim:
            events.append("claim")
            return ExperimentClaim(
                experiment_id=7,
                status="completed",
                already_completed=True,
            )

    def fail_if_run_pool_called(
        *_args: Any, **_kwargs: Any
    ) -> tuple[list[dict], str, Path]:
        raise AssertionError("run_pool should not be called for a completed experiment")

    monkeypatch.setattr(maze_runner, "ExperimentStore", FakeExperimentStore)
    monkeypatch.setattr(maze_runner, "run_pool", fail_if_run_pool_called)

    returned_id = maze_runner.run_experiment(
        name="baseline",
        params={"seed": 1},
        base_dir=tmp_path / "runs",
        database_url="postgresql://example/db",
    )

    captured = capsys.readouterr()
    assert returned_id == 7
    assert events == ["open", "claim", "close"]
    assert "Experiment 'baseline' already completed" in captured.out
    assert "skipping run" in captured.out


def test_run_experiment_closes_db_while_pool_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    open_stores = 0

    class FakeExperimentStore:
        def __init__(self, _database_url: str | None = None) -> None:
            nonlocal open_stores
            open_stores += 1
            events.append("open")

        def __enter__(self) -> "FakeExperimentStore":
            return self

        def __exit__(self, *_args: Any) -> None:
            nonlocal open_stores
            open_stores -= 1
            events.append("close")

        def claim_experiment(self, **kwargs: Any) -> ExperimentClaim:
            events.append("claim")
            assert kwargs["params"]["seed"] == 1
            assert "automaton_params" in kwargs["params"]
            assert kwargs["run_id"]
            return ExperimentClaim(experiment_id=7, status="claimed")

        def mark_experiment_running(self, _experiment_id: int) -> None:
            events.append("running")

        def ingest_run(self, _experiment_id: int, _run_dir: Path) -> int:
            events.append("ingest")
            return 3

        def mark_experiment_completed(self, _experiment_id: int) -> None:
            events.append("completed")

        def mark_experiment_failed(self, _experiment_id: int, _error: str) -> None:
            events.append("failed")

    def fake_run_pool(*_args: Any, **_kwargs: Any) -> tuple[list[dict], str, Path]:
        assert open_stores == 0
        assert _kwargs["params"]["seed"] == 1
        assert "automaton_params" in _kwargs["params"]
        assert _kwargs["run_id"]
        events.append("run_pool")
        return (
            [{"generation": 1}],
            _kwargs["run_id"],
            tmp_path / "runs" / _kwargs["run_id"],
        )

    monkeypatch.setattr(maze_runner, "ExperimentStore", FakeExperimentStore)
    monkeypatch.setattr(maze_runner, "run_pool", fake_run_pool)

    returned_id = maze_runner.run_experiment(
        name="new-baseline",
        params={"seed": 1},
        base_dir=tmp_path / "runs",
        database_url="postgresql://example/db",
    )

    assert returned_id == 7
    assert events == [
        "open",
        "claim",
        "running",
        "close",
        "run_pool",
        "open",
        "ingest",
        "completed",
        "close",
    ]


def test_run_experiment_marks_failed_when_pool_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeExperimentStore:
        def __init__(self, _database_url: str | None = None) -> None:
            events.append("open")

        def __enter__(self) -> "FakeExperimentStore":
            return self

        def __exit__(self, *_args: Any) -> None:
            events.append("close")

        def claim_experiment(self, **_kwargs: Any) -> ExperimentClaim:
            events.append("claim")
            return ExperimentClaim(experiment_id=11, status="claimed")

        def mark_experiment_running(self, _experiment_id: int) -> None:
            events.append("running")

        def mark_experiment_failed(self, experiment_id: int, error: str) -> None:
            events.append(f"failed:{experiment_id}:{error}")

    def fake_run_pool(*_args: Any, **_kwargs: Any) -> tuple[list[dict], str, Path]:
        events.append("run_pool")
        raise RuntimeError("boom")

    monkeypatch.setattr(maze_runner, "ExperimentStore", FakeExperimentStore)
    monkeypatch.setattr(maze_runner, "run_pool", fake_run_pool)

    with pytest.raises(RuntimeError, match="boom"):
        maze_runner.run_experiment(
            name="new-baseline",
            params={"seed": 1},
            base_dir=tmp_path / "runs",
            database_url="postgresql://example/db",
        )

    assert events == [
        "open",
        "claim",
        "running",
        "close",
        "run_pool",
        "open",
        "failed:11:RuntimeError: boom",
        "close",
    ]


def test_run_pool_uses_params_for_launch_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = maze_runner.default_experiment_params()
    params.update(
        {
            "total_populations": 3,
            "max_parallel": 2,
            "max_generations": 5,
            "ticks_per_restart": 7,
            "restarts_per_gen": 2,
            "population_size": 4,
            "side_length_bits": 4,
            "maze_seed": 11,
            "population_seed": 100,
            "fingerprint_enabled": True,
            "fingerprint_bits": 8,
            "fingerprint_tournament_k": 3,
            "fingerprint_mutation_rate": 0.05,
            "checkpoint_interval": 2,
            "poll_interval_s": 0.0,
            "automaton_params": {"state_bits": 6, "resp_bits": 3},
        }
    )
    launched: list[tuple[int, maze_runner.PopulationConfig]] = []

    class DoneHandle:
        def __init__(self, population_id: int) -> None:
            self.population_id = population_id

        @property
        def is_running(self) -> bool:
            return False

        @property
        def progress(self) -> dict[str, Any]:
            return {
                "generation": 5,
                "max_fitness": 1.0,
                "mean_fitness": 0.5,
                "best_max_fitness": 1.0,
                "best_mean_fitness": 0.5,
                "is_running": False,
            }

    def fake_launch_populations(
        configs: list[maze_runner.PopulationConfig],
        max_generations: int,
        base_dir: Path,
        *,
        run_id: str | None = None,
        start_pop_id: int = 0,
    ) -> list[DoneHandle]:
        assert max_generations == 5
        assert base_dir == tmp_path / "runs"
        assert run_id == "run-x"
        assert len(configs) == 1
        launched.append((start_pop_id, configs[0]))
        return [DoneHandle(start_pop_id)]

    monkeypatch.setattr(maze_runner, "launch_populations", fake_launch_populations)
    monkeypatch.setattr(maze_runner.time, "sleep", lambda _seconds: None)

    snapshots, run_id, run_dir = maze_runner.run_pool(
        base_dir=tmp_path / "runs",
        run_id="run-x",
        params=params,
    )

    assert len(snapshots) == 3
    assert run_id == "run-x"
    assert run_dir == tmp_path / "runs" / "run-x"
    assert [pop_id for pop_id, _config in launched] == [0, 1, 2]
    for pop_id, config in launched:
        assert config.size == 4
        assert config.ticks_per_restart == 7
        assert config.restarts_per_gen == 2
        assert config.seed == 100 + pop_id
        assert config.automaton_params == {"state_bits": 6, "resp_bits": 3}
        assert config.checkpoint_config is not None
        assert config.checkpoint_config.enabled is True
        assert config.checkpoint_config.generation_interval == 2
        assert config.fingerprint_config is not None
        assert config.fingerprint_config.bits == 8
        assert config.fingerprint_config.tournament_k == 3
        assert config.fingerprint_config.mutation_rate == pytest.approx(0.05)
