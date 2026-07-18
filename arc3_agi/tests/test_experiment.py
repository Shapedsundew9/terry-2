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

    returned_id = maze_runner.run_experiment1(
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
        events.append("run_pool")
        return ([{"generation": 1}], "run-1", tmp_path / "runs" / "run-1")

    monkeypatch.setattr(maze_runner, "ExperimentStore", FakeExperimentStore)
    monkeypatch.setattr(maze_runner, "run_pool", fake_run_pool)

    returned_id = maze_runner.run_experiment1(
        name="new-baseline",
        params={"seed": 1},
        base_dir=tmp_path / "runs",
        db_path=tmp_path / "runs.duckdb",
    )

    assert returned_id == 7
    assert events == ["open", "check", "close", "run_pool", "open", "create", "ingest", "close"]
