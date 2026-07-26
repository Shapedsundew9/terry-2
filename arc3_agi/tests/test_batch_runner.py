from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from arc3_agi import batch_runner
from arc3_agi.runner import PopulationConfig, PopulationHandle


class DoneHandle:
    def __init__(self, population_id: int, generation: int = 3) -> None:
        self.population_id = population_id
        self.generation = generation

    @property
    def is_running(self) -> bool:
        return False

    @property
    def progress(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "min_fitness": 0.1,
            "max_fitness": 0.9,
            "mean_fitness": 0.5,
            "best_max_fitness": 0.9,
            "best_mean_fitness": 0.5,
            "is_running": False,
        }


def _config(population_id: int) -> PopulationConfig:
    return cast(PopulationConfig, population_id)


def test_run_batch_launches_and_returns_completed_handles(tmp_path: Path) -> None:
    expected = [DoneHandle(0), DoneHandle(1)]
    launch_calls: list[tuple[list[PopulationConfig], int, Path]] = []

    def fake_launch(
        configs: list[PopulationConfig],
        max_generations: int,
        base_dir: Path,
    ) -> list[PopulationHandle]:
        launch_calls.append((configs, max_generations, base_dir))
        return cast(list[PopulationHandle], expected)

    configs = [_config(0), _config(1)]
    returned = batch_runner.run_batch(
        configs,
        max_generations=3,
        base_dir=tmp_path,
        poll_interval_s=0.0,
        launch=fake_launch,
    )

    assert returned == expected
    assert launch_calls == [(configs, 3, tmp_path)]


def test_run_batch_stops_workers_when_polling_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RunningHandle(DoneHandle):
        @property
        def is_running(self) -> bool:
            return True

    handles = cast(list[PopulationHandle], [RunningHandle(0)])
    stopped: list[list[PopulationHandle]] = []

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(batch_runner.time, "sleep", interrupt)
    monkeypatch.setattr(batch_runner, "stop_all", lambda values: stopped.append(values))

    with pytest.raises(KeyboardInterrupt):
        batch_runner.run_batch(
            [_config(0)],
            max_generations=3,
            base_dir=tmp_path,
            poll_interval_s=0.0,
            launch=lambda *_args, **_kwargs: handles,
        )

    assert stopped == [handles]


def test_run_pool_refills_slots_and_preserves_population_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launched_ids: list[int] = []
    built_ids: list[int] = []

    def config_factory(population_id: int) -> PopulationConfig:
        built_ids.append(population_id)
        return _config(population_id)

    def fake_launch(
        configs: list[PopulationConfig],
        max_generations: int,
        base_dir: Path,
        *,
        run_id: str | None = None,
        start_pop_id: int = 0,
    ) -> list[PopulationHandle]:
        assert configs == [_config(start_pop_id)]
        assert max_generations == 3
        assert base_dir == tmp_path
        assert run_id == "run-1"
        launched_ids.append(start_pop_id)
        return cast(list[PopulationHandle], [DoneHandle(start_pop_id)])

    monkeypatch.setattr(batch_runner.time, "sleep", lambda _seconds: None)

    snapshots, run_id, run_dir = batch_runner.run_pool(
        total_populations=5,
        max_parallel=2,
        max_generations=3,
        base_dir=tmp_path,
        config_factory=config_factory,
        poll_interval_s=0.0,
        run_id="run-1",
        launch=fake_launch,
    )

    assert built_ids == [0, 1, 2, 3, 4]
    assert launched_ids == [0, 1, 2, 3, 4]
    assert [snapshot["generation"] for snapshot in snapshots] == [3] * 5
    assert run_id == "run-1"
    assert run_dir == tmp_path / "run-1"


def test_run_pool_stops_active_workers_when_polling_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RunningHandle(DoneHandle):
        @property
        def is_running(self) -> bool:
            return True

    handles = cast(list[PopulationHandle], [RunningHandle(0)])
    stopped: list[list[PopulationHandle]] = []

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(batch_runner.time, "sleep", interrupt)
    monkeypatch.setattr(batch_runner, "stop_all", lambda values: stopped.append(values))

    with pytest.raises(KeyboardInterrupt):
        batch_runner.run_pool(
            total_populations=1,
            max_parallel=1,
            max_generations=3,
            base_dir=tmp_path,
            config_factory=_config,
            poll_interval_s=0.0,
            run_id="run-1",
            launch=lambda *_args, **_kwargs: handles,
        )

    assert stopped == [handles]
