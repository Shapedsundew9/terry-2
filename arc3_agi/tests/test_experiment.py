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
