"""Tests for checkpoint persistence (save/load) across GeneticCode, Automaton,
MazeAutomaton, and Population classes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from arc3_agi.automaton import AutomatonBase, AutomatonISBase
from arc3_agi.checkpoint import CheckpointConfig, genetic_code_from_dict
from arc3_agi.environment import Int1DArray
from arc3_agi.genetic_code import GeneticCodeDict, GeneticCodeList
from arc3_agi.maze import Maze, MazeAutomaton
from arc3_agi.population import Population

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_env() -> Int1DArray:
    """A minimal 1-D environment used for non-maze automaton tests."""
    env = Int1DArray("test_env", array=[0, 1, 2, 3])
    return env


@pytest.fixture()
def maze() -> Maze:
    """A small maze (side = 2**4 = 16) for MazeAutomaton tests."""
    return Maze("test_maze", side_length_bits=4, seed=42)


# ---------------------------------------------------------------------------
# Minimal concrete AutomatonBase subclass (no env_bits required)
# ---------------------------------------------------------------------------


class SimpleAutomaton(AutomatonBase):
    """Minimal concrete automaton for testing AutomatonBase checkpointing."""

    def tick(self) -> int:
        return 0


class SimpleISAutomaton(AutomatonISBase):
    """Minimal concrete automaton for testing AutomatonISBase checkpointing."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("env_bits", 2)
        kwargs.setdefault("state_bits", 2)
        kwargs.setdefault("resp_bits", 2)
        super().__init__(**kwargs)


# ---------------------------------------------------------------------------
# GeneticCodeDict — round-trip and file I/O
# ---------------------------------------------------------------------------


def test_genetic_code_dict_round_trip() -> None:
    code = GeneticCodeDict({0: 3, 5: 7, 100: 2}, seed=42, resp_bits=4)
    d = code.to_dict()
    arrays = code.to_arrays()

    assert d["type"] == "GeneticCodeDict"
    assert d["resp_bits"] == 4
    assert d["seed"] == 42

    restored = GeneticCodeDict.from_dict(d, arrays)
    assert restored.resp_bits == 4
    assert restored._seed == 42
    assert restored[0] == 3
    assert restored[5] == 7
    assert restored[100] == 2


def test_genetic_code_dict_save_load(tmp_path: Path) -> None:
    code = GeneticCodeDict({1: 1, 2: 2, 3: 3}, seed=7, resp_bits=3)
    code.save(tmp_path / "gc")

    assert (tmp_path / "gc.toml").exists()
    assert (tmp_path / "gc.npz").exists()

    restored = GeneticCodeDict.load(tmp_path / "gc")
    assert restored[1] == 1
    assert restored[2] == 2
    assert restored[3] == 3
    assert restored.resp_bits == 3


# ---------------------------------------------------------------------------
# GeneticCodeList — round-trip
# ---------------------------------------------------------------------------


def test_genetic_code_list_round_trip() -> None:
    code = GeneticCodeList([5, 3, 1, 7, 2], seed=99, resp_bits=3)
    d = code.to_dict()
    arrays = code.to_arrays()

    assert d["type"] == "GeneticCodeList"
    restored = GeneticCodeList.from_dict(d, arrays)
    assert list(restored._code) == [5, 3, 1, 7, 2]
    assert restored.resp_bits == 3
    assert restored._seed == 99


def test_genetic_code_list_save_load(tmp_path: Path) -> None:
    code = GeneticCodeList([10, 20, 30], seed=11, resp_bits=5)
    code.save(tmp_path / "gcl")
    restored = GeneticCodeList.load(tmp_path / "gcl")
    assert list(restored._code) == [10, 20, 30]


# ---------------------------------------------------------------------------
# genetic_code_from_dict factory
# ---------------------------------------------------------------------------


def test_genetic_code_from_dict_dispatches_dict() -> None:
    code = GeneticCodeDict({0: 1}, seed=1, resp_bits=1)
    restored = genetic_code_from_dict(code.to_dict(), code.to_arrays())
    assert isinstance(restored, GeneticCodeDict)


def test_genetic_code_from_dict_dispatches_list() -> None:
    code = GeneticCodeList([1, 2], seed=1, resp_bits=1)
    restored = genetic_code_from_dict(code.to_dict(), code.to_arrays())
    assert isinstance(restored, GeneticCodeList)


# ---------------------------------------------------------------------------
# AutomatonBase — round-trip
# ---------------------------------------------------------------------------


def test_automaton_base_round_trip(simple_env: Int1DArray) -> None:
    gc = GeneticCodeDict({0: 1, 1: 0}, seed=5, resp_bits=1)
    auto = SimpleAutomaton(environment=simple_env, name="bot1", genetic_code=gc)
    auto.fitness = 3.14
    auto.coords = [2]
    auto.last_action = 1

    d = auto.to_dict()
    arrays = auto.to_arrays()

    restored = SimpleAutomaton.from_dict(d, arrays, environment=simple_env)
    assert restored.name == "bot1"
    assert restored.fitness == pytest.approx(3.14)
    assert restored.coords == [2]
    assert restored.last_action == 1
    assert restored.genetic_code[0] == 1
    assert restored.genetic_code[1] == 0


def test_automaton_base_save_load(tmp_path: Path, simple_env: Int1DArray) -> None:
    gc = GeneticCodeDict({0: 1}, seed=3, resp_bits=1)
    auto = SimpleAutomaton(environment=simple_env, name="saved_bot", genetic_code=gc)
    auto.fitness = 1.0
    auto.save(tmp_path / "auto")

    assert (tmp_path / "auto.toml").exists()
    assert (tmp_path / "auto.npz").exists()

    restored = SimpleAutomaton.load(tmp_path / "auto", environment=simple_env)
    assert restored.name == "saved_bot"


# ---------------------------------------------------------------------------
# AutomatonISBase — round-trip
# ---------------------------------------------------------------------------


def test_automaton_is_base_round_trip(simple_env: Int1DArray) -> None:
    gc = GeneticCodeDict({0: 3}, seed=8, resp_bits=4)
    auto = SimpleISAutomaton(environment=simple_env, genetic_code=gc)
    auto.internal_state = 2
    auto.fitness = 7.7
    auto.coords = [1]

    d = auto.to_dict()
    arrays = auto.to_arrays()

    restored = SimpleISAutomaton.from_dict(d, arrays, environment=simple_env)
    assert restored.internal_state == 2
    assert restored.env_bits == 2
    assert restored.state_bits == 2
    assert restored.resp_bits == 2
    assert restored.fitness == pytest.approx(7.7)


# ---------------------------------------------------------------------------
# MazeAutomaton — round-trip
# ---------------------------------------------------------------------------


def test_maze_automaton_round_trip(maze: Maze) -> None:
    auto = MazeAutomaton(environment=maze)
    # Modify energy grid so we can verify it is restored exactly.
    auto.energy_grid[0] = 42
    auto.energy_grid[1] = 99
    auto.energy = 5
    auto.fitness = 12.0

    d = auto.to_dict()
    arrays = auto.to_arrays()

    restored = MazeAutomaton.from_dict(d, arrays, environment=maze)
    assert restored.energy == 5
    assert restored.fitness == pytest.approx(12.0)
    assert restored.energy_grid[0] == 42
    assert restored.energy_grid[1] == 99
    assert len(restored.energy_grid) == len(auto.energy_grid)


def test_maze_automaton_save_load(tmp_path: Path, maze: Maze) -> None:
    auto = MazeAutomaton(environment=maze)
    auto.energy = 3
    auto.save(tmp_path / "maze_auto")

    restored = MazeAutomaton.load(tmp_path / "maze_auto", environment=maze)
    assert restored.energy == 3
    assert restored._grid_width == maze.width


# ---------------------------------------------------------------------------
# Environment identity validation
# ---------------------------------------------------------------------------


def test_environment_mismatch_raises(simple_env: Int1DArray) -> None:
    gc = GeneticCodeDict({}, seed=1, resp_bits=1)
    auto = SimpleAutomaton(environment=simple_env, genetic_code=gc)
    d = auto.to_dict()
    arrays = auto.to_arrays()

    wrong_env = Int1DArray("wrong_name", array=[0])
    with pytest.raises(ValueError, match="Environment mismatch"):
        SimpleAutomaton.from_dict(d, arrays, environment=wrong_env)


# ---------------------------------------------------------------------------
# Population — round-trip
# ---------------------------------------------------------------------------


def _make_population(
    maze: Maze,
    size: int = 4,
    checkpoint_config: CheckpointConfig | None = None,
) -> Population:
    cfg = (
        checkpoint_config
        if checkpoint_config is not None
        else CheckpointConfig(enabled=False)
    )
    return Population(size, MazeAutomaton, maze, checkpoint_config=cfg)


def test_population_forwards_automaton_params_to_initial_and_offspring(
    maze: Maze,
) -> None:
    pop = Population(
        4,
        MazeAutomaton,
        maze,
        checkpoint_config=CheckpointConfig(enabled=False),
        automaton_params={"state_bits": 6, "resp_bits": 3},
    )

    assert all(isinstance(a, MazeAutomaton) and a.state_bits == 6 for a in pop.automata)
    assert all(isinstance(a, MazeAutomaton) and a.resp_bits == 3 for a in pop.automata)

    for i, automaton in enumerate(pop.automata):
        automaton.fitness = float(i + 1)
    pop.evolve()

    assert all(isinstance(a, MazeAutomaton) and a.state_bits == 6 for a in pop.automata)
    assert all(isinstance(a, MazeAutomaton) and a.resp_bits == 3 for a in pop.automata)


def test_population_round_trip_preserves_automaton_params_for_offspring(
    tmp_path: Path,
    maze: Maze,
) -> None:
    params = {"state_bits": 6, "resp_bits": 3}
    pop = Population(
        4,
        MazeAutomaton,
        maze,
        checkpoint_config=CheckpointConfig(enabled=False),
        automaton_params=params,
    )

    pop.save(tmp_path / "pop_custom_params")
    restored = Population.load(
        tmp_path / "pop_custom_params",
        environment=maze,
        AutomatonClass=MazeAutomaton,
    )

    assert restored._automaton_params == params
    for i, automaton in enumerate(restored.automata):
        automaton.fitness = float(i + 1)
    restored.evolve()

    assert all(a.state_bits == 6 for a in restored.automata)
    assert all(a.resp_bits == 3 for a in restored.automata)


def test_population_round_trip(tmp_path: Path, maze: Maze) -> None:
    pop = _make_population(maze, size=4)
    # Simulate two generations of evolution.
    for automaton in pop.automata:
        automaton.fitness = float(id(automaton) % 100)
    pop.evolve()
    for automaton in pop.automata:
        automaton.fitness = float(id(automaton) % 100) + 1.0
    pop.evolve()

    pop.save(tmp_path / "pop")
    assert (tmp_path / "pop.toml").exists()
    assert (tmp_path / "pop.npz").exists()

    restored = Population.load(
        tmp_path / "pop",
        environment=maze,
        AutomatonClass=MazeAutomaton,
    )
    assert restored.generation == 2
    assert restored.tick_count == 0
    assert len(restored.fitness_history) == 2
    assert len(restored.automata) == 4
    # Full fitnesses arrays must survive the NPZ round-trip.
    for entry in restored.fitness_history:
        assert len(entry["fitnesses"]) == 4
        assert all(math.isfinite(f) for f in entry["fitnesses"])
        assert entry["duration_s"] is not None
        assert math.isfinite(entry["duration_s"])
    # Verify automata were reconstructed (energy_grid size matches).
    for a in restored.automata:
        assert len(a.energy_grid) == maze.width * maze.height


# ---------------------------------------------------------------------------
# Checkpoint files — every generation (interval = 1)
# ---------------------------------------------------------------------------


def test_checkpoint_file_pair_created_per_generation(maze: Maze) -> None:
    cfg = CheckpointConfig(enabled=True, base_dir=Path("runs"), generation_interval=1)
    pop = Population(4, MazeAutomaton, maze, checkpoint_config=cfg)
    assert pop._run_dir is not None

    for a in pop.automata:
        a.fitness = 1.0
    pop.evolve()

    assert (pop._run_dir / "gen_000001.toml").exists()
    assert (pop._run_dir / "gen_000001.npz").exists()

    # Cleanup.
    import shutil

    shutil.rmtree(cfg.base_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Checkpoint files — every N generations (interval > 1)
# ---------------------------------------------------------------------------


def test_checkpoint_file_pair_created_per_n_generations(maze: Maze) -> None:
    """With generation_interval=3, files appear at gen 3 and 6 but not 1, 2, 4, 5."""
    cfg = CheckpointConfig(enabled=True, base_dir=Path("runs"), generation_interval=3)
    pop = Population(2, MazeAutomaton, maze, checkpoint_config=cfg)
    assert pop._run_dir is not None

    for gen in range(6):
        for a in pop.automata:
            a.fitness = float(gen)
        pop.evolve()

    assert (pop._run_dir / "gen_000003.toml").exists()
    assert (pop._run_dir / "gen_000006.toml").exists()
    assert not (pop._run_dir / "gen_000001.toml").exists()
    assert not (pop._run_dir / "gen_000002.toml").exists()

    import shutil

    shutil.rmtree(cfg.base_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Checkpoint disabled
# ---------------------------------------------------------------------------


def test_checkpoint_disabled(maze: Maze) -> None:
    cfg = CheckpointConfig(enabled=False)
    pop = Population(2, MazeAutomaton, maze, checkpoint_config=cfg)
    assert pop._run_dir is None

    for a in pop.automata:
        a.fitness = 1.0
    pop.evolve()
    for _ in range(10):
        pop.tick()


# ---------------------------------------------------------------------------
# Fitness history accumulation
# ---------------------------------------------------------------------------


def test_fitness_history_accumulated(maze: Maze) -> None:
    pop = _make_population(maze, size=4)
    for gen in range(3):
        for a in pop.automata:
            a.fitness = float(gen + 1)
        pop.evolve()

    assert len(pop.fitness_history) == 3
    for i, entry in enumerate(pop.fitness_history):
        assert entry["generation"] == i + 1
        assert math.isfinite(entry["min_fitness"])
        assert math.isfinite(entry["max_fitness"])
        assert math.isfinite(entry["mean_fitness"])
        assert entry["duration_s"] is not None
        assert math.isfinite(entry["duration_s"])
        assert len(entry["fitnesses"]) == 4


# ---------------------------------------------------------------------------
# Restored population creates a new run dir
# ---------------------------------------------------------------------------


def test_population_load_creates_new_run_dir(tmp_path: Path, maze: Maze) -> None:
    cfg = CheckpointConfig(enabled=True, base_dir=tmp_path / "runs")
    pop = Population(2, MazeAutomaton, maze, checkpoint_config=cfg)
    original_run_dir = pop._run_dir

    pop.save(tmp_path / "pop_ckpt")

    restored = Population.load(
        tmp_path / "pop_ckpt",
        environment=maze,
        AutomatonClass=MazeAutomaton,
    )
    assert restored._run_dir != original_run_dir
