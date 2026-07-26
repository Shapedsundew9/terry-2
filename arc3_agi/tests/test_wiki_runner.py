from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from arc3_agi import wiki_runner
from arc3_agi.environment import ByteEnv
from arc3_agi.runner import PopulationConfig
from arc3_agi.wiki_text_2 import WikiAutomaton


class DoneHandle:
    def __init__(self, population_id: int, generation: int) -> None:
        self.population_id = population_id
        self.generation = generation

    @property
    def is_running(self) -> bool:
        return False

    @property
    def progress(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "min_fitness": 0.0,
            "max_fitness": 0.5,
            "mean_fitness": 0.25,
            "best_max_fitness": 0.5,
            "best_mean_fitness": 0.25,
            "is_running": False,
        }


def test_default_params_identify_dataset_and_automaton() -> None:
    params = wiki_runner.default_experiment_params()

    assert params["dataset_name"] == "Salesforce/wikitext"
    assert params["dataset_config"] == "wikitext-2-raw-v1"
    assert params["dataset_split"] == "train"
    assert params["ticks_per_restart"] == 1000
    assert params["restarts_per_gen"] == 1
    assert params["fingerprint_enabled"] is True
    assert params["automaton_params"] == {
        "env_bits": 16,
        "state_bits": 8,
        "resp_bits": 8,
        "num_clauses": 16,
    }


def test_resolve_params_merges_automaton_overrides() -> None:
    resolved = wiki_runner._resolve_experiment_params(
        {
            "max_generations": 7,
            "automaton_params": {"state_bits": 5, "num_clauses": 8},
        }
    )

    assert resolved["max_generations"] == 7
    assert resolved["automaton_params"] == {
        "env_bits": 16,
        "state_bits": 5,
        "resp_bits": 8,
        "num_clauses": 8,
    }


def test_build_config_uses_wiki_automaton_and_population_seed() -> None:
    environment = ByteEnv(name="test", array=["abc"])
    params = wiki_runner.default_experiment_params()
    params.update(
        {
            "population_size": 4,
            "ticks_per_restart": 7,
            "restarts_per_gen": 2,
            "population_seed": 100,
            "checkpoint_interval": 3,
            "fingerprint_bits": 8,
            "fingerprint_tournament_k": 3,
            "fingerprint_mutation_rate": 0.05,
        }
    )

    config = wiki_runner._build_config(2, environment, params)

    assert config.size == 4
    assert config.AutomatonClass is WikiAutomaton
    assert config.environment is environment
    assert config.ticks_per_restart == 7
    assert config.restarts_per_gen == 2
    assert config.seed == 102
    assert config.checkpoint_config is not None
    assert config.checkpoint_config.enabled is True
    assert config.checkpoint_config.generation_interval == 3
    assert config.fingerprint_config is not None
    assert config.fingerprint_config.bits == 8
    assert config.fingerprint_config.tournament_k == 3
    assert config.fingerprint_config.mutation_rate == pytest.approx(0.05)


def test_run_delegates_fixed_batch_with_loaded_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = ByteEnv(name="fixed", array=["abc"])
    captured: dict[str, Any] = {}
    handles = [DoneHandle(0, 5), DoneHandle(1, 5)]

    def fake_run_batch(configs, **kwargs):
        captured["configs"] = configs
        captured.update(kwargs)
        return handles

    monkeypatch.setattr(
        wiki_runner,
        "load_wikitext_environment",
        lambda *_args: environment,
    )
    monkeypatch.setattr(wiki_runner, "_run_batch", fake_run_batch)

    returned = wiki_runner.run(
        base_dir=tmp_path,
        params={
            "max_parallel": 2,
            "max_generations": 5,
            "ticks_per_restart": 7,
            "restarts_per_gen": 1,
            "population_size": 4,
            "checkpoint_interval": 0,
            "poll_interval_s": 0.0,
        },
    )

    configs = captured["configs"]
    assert returned == handles
    assert [config.seed for config in configs] == [0, 1]
    assert all(config.environment is environment for config in configs)
    assert captured["max_generations"] == 5
    assert captured["base_dir"] == tmp_path
    assert captured["poll_interval_s"] == 0.0
    assert captured["launch"] is wiki_runner.launch_populations


def test_run_pool_forwards_dataset_and_builds_each_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = ByteEnv(name="custom", array=["abc"])
    loaded: list[tuple[str, str, str]] = []
    launched: list[tuple[int, PopulationConfig]] = []

    def fake_loader(name: str, config: str, split: str) -> ByteEnv:
        loaded.append((name, config, split))
        return environment

    def fake_launch(
        configs: list[PopulationConfig],
        max_generations: int,
        base_dir: Path,
        *,
        run_id: str | None = None,
        start_pop_id: int = 0,
    ) -> list[DoneHandle]:
        assert len(configs) == 1
        assert max_generations == 5
        assert base_dir == tmp_path
        assert run_id == "wiki-run"
        launched.append((start_pop_id, configs[0]))
        return [DoneHandle(start_pop_id, max_generations)]

    monkeypatch.setattr(wiki_runner, "load_wikitext_environment", fake_loader)
    monkeypatch.setattr(wiki_runner, "launch_populations", fake_launch)
    params = {
        "total_populations": 3,
        "max_parallel": 2,
        "max_generations": 5,
        "ticks_per_restart": 7,
        "restarts_per_gen": 2,
        "population_size": 4,
        "population_seed": 20,
        "checkpoint_interval": 0,
        "poll_interval_s": 0.0,
        "dataset_name": "example/wiki",
        "dataset_config": "raw",
        "dataset_split": "validation",
        "automaton_params": {"state_bits": 6},
    }

    snapshots, run_id, run_dir = wiki_runner.run_pool(
        base_dir=tmp_path,
        run_id="wiki-run",
        params=params,
    )

    assert loaded == [("example/wiki", "raw", "validation")]
    assert [population_id for population_id, _ in launched] == [0, 1, 2]
    assert [config.seed for _, config in launched] == [20, 21, 22]
    assert all(config.environment is environment for _, config in launched)
    assert all(config.AutomatonClass is WikiAutomaton for _, config in launched)
    assert all(config.automaton_params["state_bits"] == 6 for _, config in launched)
    assert len(snapshots) == 3
    assert run_id == "wiki-run"
    assert run_dir == tmp_path / "wiki-run"


def test_run_experiment_resolves_params_before_tracking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run_tracked_experiment(**kwargs) -> int:
        captured.update(kwargs)
        return 9

    monkeypatch.setattr(
        wiki_runner, "_run_tracked_experiment", fake_run_tracked_experiment
    )

    experiment_id = wiki_runner.run_experiment(
        name="wiki-test",
        params={"automaton_params": {"state_bits": 3}},
        description="test",
        base_dir=tmp_path,
        database_url="postgresql://example/db",
    )

    assert experiment_id == 9
    assert captured["params"]["automaton_params"]["state_bits"] == 3
    assert captured["params"]["automaton_params"]["env_bits"] == 16
    assert captured["pool_runner"] is wiki_runner.run_pool
    assert captured["store_factory"] is wiki_runner.ExperimentStore


def test_main_runs_named_tracked_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        wiki_runner,
        "run_experiment",
        lambda **kwargs: calls.append(kwargs) or 1,
    )

    wiki_runner.main()

    assert len(calls) == 1
    assert calls[0]["name"] == "wikitext2-baseline-state8-tsetlin"
    assert calls[0]["params"] == wiki_runner.default_experiment_params()
    assert "WikiText 2" in calls[0]["description"]


def test_run_pool_executes_tiny_wiki_population_in_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = ByteEnv(name="smoke", array=["abc", "def"])
    monkeypatch.setattr(
        wiki_runner,
        "load_wikitext_environment",
        lambda *_args: environment,
    )
    params = {
        "total_populations": 1,
        "max_parallel": 1,
        "max_generations": 1,
        "ticks_per_restart": 3,
        "restarts_per_gen": 1,
        "population_size": 4,
        "population_seed": 0,
        "fingerprint_enabled": False,
        "checkpoint_interval": 0,
        "poll_interval_s": 0.01,
    }

    snapshots, _, run_dir = wiki_runner.run_pool(
        base_dir=tmp_path,
        run_id="smoke-run",
        params=params,
    )

    assert snapshots[0]["generation"] == 1
    assert "error" not in snapshots[0]
    history_path = run_dir / "pop_0" / "fitness_history.json"
    history = json.loads(history_path.read_text())
    assert history["pop_id"] == 0
    assert [entry["generation"] for entry in history["history"]] == [1]
