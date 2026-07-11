"""Tests that a fixed seed produces fully deterministic, reproducible evolution.

The canonical reproducibility test:
  1. Run two Population instances with the *same* seed and config.
  2. Assert their complete per-generation fitness histories are identical.
  3. Run a third instance with a *different* seed.
  4. Assert its history diverges from the first two (confirming the seed matters).

All tests run Population directly (not via subprocess) so they are fast and
don't depend on multiprocessing behaviour.
"""

from __future__ import annotations

from arc3_agi.checkpoint import CheckpointConfig
from arc3_agi.fingerprint import FingerprintConfig
from arc3_agi.maze import Maze, MazeAutomaton
from arc3_agi.population import Population

# ---------------------------------------------------------------------------
# Constants — kept small so tests run in well under a second.
# ---------------------------------------------------------------------------

SIDE_BITS = 4  # 16×16 maze
POP_SIZE = 10
TICKS_PER_RESTART = 20
NUM_GENERATIONS = 5
SEED_A = 1234
SEED_B = 5678  # different seed — must produce different results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_maze(seed: int = 0) -> Maze:
    return Maze(name="repro_maze", side_length_bits=SIDE_BITS, seed=seed)


def _run_population(seed: int | None, maze: Maze) -> list[dict]:
    """Return the full fitness_history after NUM_GENERATIONS of evolution."""
    pop = Population(
        size=POP_SIZE,
        AutomatonClass=MazeAutomaton,
        environment=maze,
        checkpoint_config=CheckpointConfig(enabled=False),
        fingerprint_config=FingerprintConfig(bits=4, tournament_k=2),
        seed=seed,
    )
    for _ in range(NUM_GENERATIONS):
        pop.run_generation(TICKS_PER_RESTART)
        pop.evolve()
    return pop.fitness_history


def _history_key(history: list[dict]) -> list[tuple]:
    """Reduce fitness_history to a comparable tuple sequence."""
    return [(rec["generation"], tuple(rec["fitnesses"])) for rec in history]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Same seed → identical results; different seed → different results."""

    def test_same_seed_same_fitness_history(self) -> None:
        """Two populations seeded identically must evolve identically."""
        maze = _make_maze(seed=0)

        history_1 = _run_population(SEED_A, maze)
        history_2 = _run_population(SEED_A, maze)

        assert len(history_1) == NUM_GENERATIONS
        assert len(history_2) == NUM_GENERATIONS

        for gen_idx, (rec1, rec2) in enumerate(zip(history_1, history_2), start=1):
            assert rec1["fitnesses"] == rec2["fitnesses"], (
                f"Generation {gen_idx}: fitness lists differ between two runs "
                f"with the same seed={SEED_A}.\n"
                f"  Run 1: {rec1['fitnesses']}\n"
                f"  Run 2: {rec2['fitnesses']}"
            )

    def test_different_seeds_produce_different_histories(self) -> None:
        """Two populations with different seeds should not evolve identically."""
        maze = _make_maze(seed=0)

        history_a = _history_key(_run_population(SEED_A, maze))
        history_b = _history_key(_run_population(SEED_B, maze))

        assert history_a != history_b, (
            f"Populations with different seeds ({SEED_A} vs {SEED_B}) produced "
            "identical fitness histories — the seed has no effect."
        )

    def test_same_seed_same_initial_coords(self) -> None:
        """Automata created with the same seed must start at the same positions."""
        maze = _make_maze(seed=0)

        pop_1 = Population(
            size=POP_SIZE,
            AutomatonClass=MazeAutomaton,
            environment=maze,
            checkpoint_config=CheckpointConfig(enabled=False),
            seed=SEED_A,
        )
        pop_2 = Population(
            size=POP_SIZE,
            AutomatonClass=MazeAutomaton,
            environment=maze,
            checkpoint_config=CheckpointConfig(enabled=False),
            seed=SEED_A,
        )

        coords_1 = [list(a.coords) for a in pop_1.automata]
        coords_2 = [list(a.coords) for a in pop_2.automata]
        assert coords_1 == coords_2, (
            "Initial automaton coordinates differ between two populations "
            f"created with the same seed={SEED_A}."
        )

    def test_same_seed_same_initial_fingerprints(self) -> None:
        """Fingerprints initialised with the same seed must be identical."""
        maze = _make_maze(seed=0)
        fp_cfg = FingerprintConfig(bits=8, tournament_k=2)

        pop_1 = Population(
            size=POP_SIZE,
            AutomatonClass=MazeAutomaton,
            environment=maze,
            checkpoint_config=CheckpointConfig(enabled=False),
            fingerprint_config=fp_cfg,
            seed=SEED_A,
        )
        pop_2 = Population(
            size=POP_SIZE,
            AutomatonClass=MazeAutomaton,
            environment=maze,
            checkpoint_config=CheckpointConfig(enabled=False),
            fingerprint_config=fp_cfg,
            seed=SEED_A,
        )

        fps_1 = [a.fingerprint.value for a in pop_1.automata]
        fps_2 = [a.fingerprint.value for a in pop_2.automata]
        assert fps_1 == fps_2, (
            "Initial fingerprint values differ between two populations "
            f"created with the same seed={SEED_A}."
        )

    def test_unseeded_populations_are_accepted(self) -> None:
        """seed=None must still run without errors (non-deterministic but valid)."""
        maze = _make_maze(seed=0)
        history = _run_population(None, maze)
        assert len(history) == NUM_GENERATIONS
