"""Tests for arc3_agi.runner — parallel background population runner."""

from __future__ import annotations

import time
import multiprocessing
from pathlib import Path

import pytest

from arc3_agi.checkpoint import CheckpointConfig
from arc3_agi.fingerprint import FingerprintConfig
from arc3_agi.maze import Maze, MazeAutomaton
from arc3_agi.runner import (
    PopulationConfig,
    PopulationHandle,
    launch_populations,
    stop_all,
    wait_all,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SIDE_BITS = 4  # 16×16 maze — fast to generate and traverse
MAX_GEN = 3
TICKS = 10
POP_SIZE = 10


def _wait_forever(event: multiprocessing.Event) -> None:
    event.wait()


@pytest.fixture
def maze() -> Maze:
    return Maze(name="test_maze", side_length_bits=SIDE_BITS, seed=0)


@pytest.fixture
def basic_config(maze: Maze, tmp_path: Path) -> PopulationConfig:
    return PopulationConfig(
        size=POP_SIZE,
        AutomatonClass=MazeAutomaton,
        environment=maze,
        ticks_per_restart=TICKS,
        checkpoint_config=CheckpointConfig(
            enabled=True,
            base_dir=tmp_path / "ckpts",
            generation_interval=1,
        ),
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestPopulationConfig:
    def test_fields_accessible(self, maze: Maze) -> None:
        cfg = PopulationConfig(
            size=50,
            AutomatonClass=MazeAutomaton,
            environment=maze,
            ticks_per_restart=20,
        )
        assert cfg.size == 50
        assert cfg.AutomatonClass is MazeAutomaton
        assert cfg.ticks_per_restart == 20
        assert cfg.checkpoint_config is None
        assert cfg.fingerprint_config is None

    def test_optional_fingerprint(self, maze: Maze) -> None:
        fp = FingerprintConfig(bits=4, tournament_k=2)
        cfg = PopulationConfig(
            size=10,
            AutomatonClass=MazeAutomaton,
            environment=maze,
            ticks_per_restart=5,
            fingerprint_config=fp,
        )
        assert cfg.fingerprint_config is fp


class TestPopulationHandle:
    def test_initial_progress(self, maze: Maze, tmp_path: Path) -> None:
        """A brand-new handle returns a generation-0 dict before any work."""
        import multiprocessing

        q: multiprocessing.Queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=lambda: None, daemon=True)
        h = PopulationHandle(0, p, q)
        prog = h.progress
        assert prog["generation"] == 0
        assert prog["is_running"] is True
        assert "min_fitness" in prog
        assert "max_fitness" in prog
        assert "mean_fitness" in prog

    def test_progress_keys(
        self, basic_config: PopulationConfig, tmp_path: Path
    ) -> None:
        """Progress dict always contains the five expected keys."""
        handles = launch_populations(
            [basic_config], max_generations=MAX_GEN, base_dir=tmp_path
        )
        wait_all(handles)
        prog = handles[0].progress
        for key in (
            "generation",
            "min_fitness",
            "max_fitness",
            "mean_fitness",
            "is_running",
        ):
            assert key in prog, f"Missing key: {key}"


class TestLaunchPopulations:
    def test_returns_one_handle_per_config(
        self, basic_config: PopulationConfig, tmp_path: Path
    ) -> None:
        handles = launch_populations(
            [basic_config, basic_config], max_generations=1, base_dir=tmp_path
        )
        assert len(handles) == 2
        assert handles[0].population_id == 0
        assert handles[1].population_id == 1
        wait_all(handles)

    def test_returns_immediately(
        self, basic_config: PopulationConfig, tmp_path: Path
    ) -> None:
        """launch_populations() should return well before populations finish."""
        t0 = time.monotonic()
        handles = launch_populations(
            [basic_config], max_generations=50, base_dir=tmp_path
        )
        elapsed = time.monotonic() - t0
        # Should return in much less than 1 s even though the population runs longer.
        assert (
            elapsed < 2.0
        ), f"launch_populations took {elapsed:.2f}s — not returning quickly"
        wait_all(handles)

    def test_populations_finish(
        self, basic_config: PopulationConfig, tmp_path: Path
    ) -> None:
        handles = launch_populations(
            [basic_config, basic_config], max_generations=MAX_GEN, base_dir=tmp_path
        )
        wait_all(handles, timeout=60)
        for h in handles:
            assert not h.is_running, f"pop {h.population_id} is still running"

    def test_progress_reaches_max_generation(
        self, basic_config: PopulationConfig, tmp_path: Path
    ) -> None:
        handles = launch_populations(
            [basic_config], max_generations=MAX_GEN, base_dir=tmp_path
        )
        wait_all(handles, timeout=60)
        prog = handles[0].progress
        assert (
            prog["generation"] == MAX_GEN
        ), f"Expected generation={MAX_GEN}, got {prog['generation']}"

    def test_checkpoint_dirs_isolated(
        self, basic_config: PopulationConfig, tmp_path: Path
    ) -> None:
        """Each population writes checkpoints to a separate pop_N subdirectory."""
        base = tmp_path / "runs"
        handles = launch_populations(
            [basic_config, basic_config], max_generations=MAX_GEN, base_dir=base
        )
        wait_all(handles, timeout=60)

        # The runner creates base/<run_id>/pop_0/ and base/<run_id>/pop_1/.
        pop_dirs = sorted(base.glob("*/pop_*"))
        assert len(pop_dirs) == 2, f"Expected 2 pop dirs, found: {pop_dirs}"

        # Each pop dir must contain at least one checkpoint (.toml) file.
        for pop_dir in pop_dirs:
            toml_files = list(pop_dir.rglob("*.toml"))
            assert toml_files, f"No checkpoint .toml files found under {pop_dir}"

    def test_no_checkpoint_when_disabled(self, maze: Maze, tmp_path: Path) -> None:
        cfg = PopulationConfig(
            size=POP_SIZE,
            AutomatonClass=MazeAutomaton,
            environment=maze,
            ticks_per_restart=TICKS,
            checkpoint_config=CheckpointConfig(
                enabled=False, base_dir=tmp_path / "ckpts"
            ),
        )
        base = tmp_path / "runs"
        handles = launch_populations([cfg], max_generations=MAX_GEN, base_dir=base)
        wait_all(handles, timeout=60)
        toml_files = list(base.rglob("*.toml"))
        assert not toml_files, f"Expected no checkpoints, found: {toml_files}"

    def test_heterogeneous_configs(self, tmp_path: Path) -> None:
        """Two populations with different seeds/environments run without interference."""
        maze_a = Maze(name="maze_a", side_length_bits=SIDE_BITS, seed=1)
        maze_b = Maze(name="maze_b", side_length_bits=SIDE_BITS, seed=2)
        cfg_a = PopulationConfig(
            size=POP_SIZE,
            AutomatonClass=MazeAutomaton,
            environment=maze_a,
            ticks_per_restart=TICKS,
            checkpoint_config=CheckpointConfig(enabled=False),
        )
        cfg_b = PopulationConfig(
            size=POP_SIZE * 2,
            AutomatonClass=MazeAutomaton,
            environment=maze_b,
            ticks_per_restart=TICKS * 2,
            checkpoint_config=CheckpointConfig(enabled=False),
        )
        handles = launch_populations(
            [cfg_a, cfg_b], max_generations=MAX_GEN, base_dir=tmp_path
        )
        wait_all(handles, timeout=60)
        for h in handles:
            assert not h.is_running
            assert h.progress["generation"] == MAX_GEN

    def test_with_fingerprint_config(self, maze: Maze, tmp_path: Path) -> None:
        cfg = PopulationConfig(
            size=POP_SIZE,
            AutomatonClass=MazeAutomaton,
            environment=maze,
            ticks_per_restart=TICKS,
            checkpoint_config=CheckpointConfig(enabled=False),
            fingerprint_config=FingerprintConfig(bits=4, tournament_k=2),
        )
        handles = launch_populations([cfg], max_generations=MAX_GEN, base_dir=tmp_path)
        wait_all(handles, timeout=60)
        assert handles[0].progress["generation"] == MAX_GEN


class TestWaitAll:
    def test_blocks_until_done(
        self, basic_config: PopulationConfig, tmp_path: Path
    ) -> None:
        handles = launch_populations(
            [basic_config, basic_config], max_generations=MAX_GEN, base_dir=tmp_path
        )
        wait_all(handles, timeout=60)
        assert all(not h.is_running for h in handles)

    def test_empty_list(self) -> None:
        """wait_all on an empty list should not raise."""
        wait_all([])


class TestStopAll:
    def test_terminates_running_processes(self) -> None:
        event = multiprocessing.Event()
        queue: multiprocessing.Queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=_wait_forever,
            args=(event,),
            daemon=True,
        )
        process.start()
        handle = PopulationHandle(0, process, queue)

        stop_all([handle], timeout=1.0)

        assert not handle.is_running
