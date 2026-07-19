"""Tests for experiment database helpers and runner entry points."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from arc3_agi import maze_runner
from arc3_agi.experiment import ExperimentStore


def test_get_experiment_id_by_name_returns_existing_id(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.duckdb"
    with ExperimentStore(db_path) as store:
        existing_id = store.create_experiment(name="baseline", params={"seed": 1})

        assert store.get_experiment_id_by_name("baseline") == existing_id
        assert store.get_experiment_id_by_name("missing") is None


def test_run_experiment1_skips_existing_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db_path = tmp_path / "runs.duckdb"
    with ExperimentStore(db_path) as store:
        existing_id = store.create_experiment(name="baseline", params={"seed": 1})

    def fail_if_run_pool_called(
        *_args: Any, **_kwargs: Any
    ) -> tuple[list[dict], str, Path]:
        raise AssertionError("run_pool should not be called for an existing experiment")

    monkeypatch.setattr(maze_runner, "run_pool", fail_if_run_pool_called)

    returned_id = maze_runner.run_experiment(
        name="baseline",
        params={"seed": 1},
        base_dir=tmp_path / "runs",
        db_path=db_path,
    )

    captured = capsys.readouterr()
    assert returned_id == existing_id
    assert "Experiment 'baseline' already exists" in captured.out
    assert "skipping run" in captured.out


def test_run_experiment1_closes_db_while_pool_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    open_stores = 0

    class FakeExperimentStore:
        def __init__(self, _db_path: Path) -> None:
            nonlocal open_stores
            open_stores += 1
            events.append("open")

        def __enter__(self) -> "FakeExperimentStore":
            return self

        def __exit__(self, *_args: Any) -> None:
            nonlocal open_stores
            open_stores -= 1
            events.append("close")

        def get_experiment_id_by_name(self, _name: str) -> int | None:
            events.append("check")
            return None

        def create_experiment(self, **_kwargs: Any) -> int:
            events.append("create")
            return 7

        def ingest_run(self, _experiment_id: int, _run_dir: Path) -> int:
            events.append("ingest")
            return 3

    def fake_run_pool(*_args: Any, **_kwargs: Any) -> tuple[list[dict], str, Path]:
        assert open_stores == 0
        assert _kwargs["params"]["seed"] == 1
        assert "automaton_params" in _kwargs["params"]
        events.append("run_pool")
        return ([{"generation": 1}], "run-1", tmp_path / "runs" / "run-1")

    monkeypatch.setattr(maze_runner, "ExperimentStore", FakeExperimentStore)
    monkeypatch.setattr(maze_runner, "run_pool", fake_run_pool)

    returned_id = maze_runner.run_experiment(
        name="new-baseline",
        params={"seed": 1},
        base_dir=tmp_path / "runs",
        db_path=tmp_path / "runs.duckdb",
    )

    assert returned_id == 7
    assert events == [
        "open",
        "check",
        "close",
        "run_pool",
        "open",
        "create",
        "ingest",
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
