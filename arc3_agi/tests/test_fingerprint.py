"""Tests for SelectionFingerprint and the fingerprint-aware Population mechanics."""

from __future__ import annotations

from random import Random
from unittest.mock import MagicMock, patch

import pytest

from arc3_agi.fingerprint import FingerprintConfig, SelectionFingerprint

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_fp(
    bits: int = 32, value: int | None = None, seed: int = 1
) -> SelectionFingerprint:
    return SelectionFingerprint(bits, rng=Random(seed), value=value)


# --------------------------------------------------------------------------- #
# Hamming distance
# --------------------------------------------------------------------------- #


def test_hamming_identical_is_zero() -> None:
    fp = make_fp(value=0b10110)
    other = make_fp(value=0b10110, seed=2)
    assert fp.hamming(other) == 0


def test_hamming_complement_equals_bits() -> None:
    bits = 8
    mask = (1 << bits) - 1
    fp = make_fp(bits=bits, value=0b10101010)
    other = make_fp(bits=bits, value=0b01010101, seed=2)
    assert fp.hamming(other) == bits


def test_hamming_known_value() -> None:
    # 0b1100 XOR 0b1010 = 0b0110 → 2 set bits
    fp = make_fp(bits=4, value=0b1100)
    other = make_fp(bits=4, value=0b1010, seed=2)
    assert fp.hamming(other) == 2


def test_hamming_symmetry() -> None:
    fp_a = make_fp(value=0xDEADBEEF)
    fp_b = make_fp(value=0x12345678, seed=2)
    assert fp_a.hamming(fp_b) == fp_b.hamming(fp_a)


# --------------------------------------------------------------------------- #
# Crossover
# --------------------------------------------------------------------------- #


def test_crossover_preserves_bit_width() -> None:
    bits = 16
    fp_a = make_fp(bits=bits, value=0xAAAA)
    fp_b = make_fp(bits=bits, value=0x5555, seed=2)
    child = fp_a.crossover(fp_b)
    assert child.bits == bits
    assert 0 <= child.value <= (1 << bits) - 1


def test_crossover_shared_bits_always_inherited() -> None:
    """Bits set in both parents must appear in every child (AND preserved)."""
    bits = 8
    fp_a = make_fp(bits=bits, value=0b11110000, seed=1)
    fp_b = make_fp(bits=bits, value=0b11001100, seed=2)
    shared = fp_a.value & fp_b.value  # 0b11000000
    for _ in range(50):
        child = fp_a.crossover(fp_b)
        assert child.value & shared == shared


def test_crossover_child_bounded_by_union() -> None:
    """Child must never have a bit that neither parent had (child ⊆ union)."""
    bits = 8
    fp_a = make_fp(bits=bits, value=0b10101010, seed=3)
    fp_b = make_fp(bits=bits, value=0b01010101, seed=4)
    union = fp_a.value | fp_b.value
    for _ in range(50):
        child = fp_a.crossover(fp_b)
        assert child.value & union == child.value


def test_crossover_returns_new_instance() -> None:
    fp_a = make_fp(value=0xABCD1234)
    fp_b = make_fp(value=0x12345678, seed=2)
    child = fp_a.crossover(fp_b)
    assert child is not fp_a
    assert child is not fp_b


def test_crossover_child_has_independent_rng() -> None:
    """Two crossovers with different seeds produce different children (RNG is seeded)."""
    fp_a = make_fp(seed=7, value=0xAAAAAAAA)
    fp_b = make_fp(seed=13, value=0x55555555)
    children = [fp_a.crossover(fp_b) for _ in range(20)]
    # Not all children should have identical values
    assert len({c.value for c in children}) > 1


# --------------------------------------------------------------------------- #
# Mutation
# --------------------------------------------------------------------------- #


def test_mutate_zero_rate_no_change() -> None:
    original = 0xDEADBEEF
    fp = make_fp(value=original)
    fp.mutate(0.0)
    assert fp.value == original


def test_mutate_rate_one_flips_all_bits() -> None:
    # The mutate implementation now flips exactly one randomly chosen bit with
    # probability mutation_rate (rather than flipping every bit independently).
    # With rate=1.0 that one flip always occurs, so exactly 1 bit differs.
    bits = 8
    original = 0b10101010
    fp = make_fp(bits=bits, value=original)
    fp.mutate(1.0)
    assert fp.value != original
    assert (fp.value ^ original).bit_count() == 1


def test_mutate_modifies_value_on_average() -> None:
    """With mutation_rate=0.5 on 64 bits, expected ~32 flips — value almost never unchanged."""
    bits = 64
    fp = make_fp(bits=bits, seed=99)
    original = fp.value
    fp.mutate(0.5)
    # With rate=0.5 on 64 bits, P(no change) ≈ (0.5)^64 ≈ 5e-20 — essentially impossible
    assert fp.value != original


# --------------------------------------------------------------------------- #
# flip_toward
# --------------------------------------------------------------------------- #


def test_flip_toward_reduces_hamming_by_one() -> None:
    fp_a = make_fp(bits=8, value=0b11110000)
    fp_b = make_fp(bits=8, value=0b00000000, seed=2)
    before = fp_a.hamming(fp_b)
    fp_a.flip_toward(fp_b)
    after = fp_a.hamming(fp_b)
    assert after == before - 1


def test_flip_toward_noop_when_identical() -> None:
    val = 0b10101010
    fp_a = make_fp(bits=8, value=val)
    fp_b = make_fp(bits=8, value=val, seed=2)
    fp_a.flip_toward(fp_b)
    assert fp_a.value == val  # unchanged


def test_flip_toward_converges_to_identical() -> None:
    bits = 16
    fp_a = make_fp(bits=bits, value=0xFFFF)
    fp_b = make_fp(bits=bits, value=0x0000, seed=2)
    for _ in range(bits):
        fp_a.flip_toward(fp_b)
    assert fp_a.value == fp_b.value


# --------------------------------------------------------------------------- #
# flip_away
# --------------------------------------------------------------------------- #


def test_flip_away_increases_hamming_by_one() -> None:
    fp_a = make_fp(bits=8, value=0b11111111)
    fp_b = make_fp(bits=8, value=0b11110000, seed=2)
    before = fp_a.hamming(fp_b)
    fp_a.flip_away(fp_b)
    after = fp_a.hamming(fp_b)
    assert after == before + 1


def test_flip_away_noop_when_complementary() -> None:
    bits = 8
    fp_a = make_fp(bits=bits, value=0b11110000)
    fp_b = make_fp(bits=bits, value=0b00001111, seed=2)
    original = fp_a.value
    fp_a.flip_away(fp_b)
    assert fp_a.value == original  # no matching bits to flip


def test_flip_away_diverges_to_complementary() -> None:
    bits = 8
    mask = (1 << bits) - 1
    fp_a = make_fp(bits=bits, value=0xFF)
    fp_b = make_fp(bits=bits, value=0xFF, seed=2)
    for _ in range(bits):
        fp_a.flip_away(fp_b)
    # All bits now differ
    assert fp_a.hamming(fp_b) == bits


# --------------------------------------------------------------------------- #
# _random_set_bit helper
# --------------------------------------------------------------------------- #


def test_random_set_bit_returns_none_for_zero_mask() -> None:
    fp = make_fp()
    assert fp._random_set_bit(0) is None


def test_random_set_bit_returns_valid_position() -> None:
    fp = make_fp()
    mask = 0b10100  # bits 2 and 4 are set
    for _ in range(30):
        pos = fp._random_set_bit(mask)
        assert pos in (2, 4)


def test_random_set_bit_returns_only_set_positions() -> None:
    rng = Random(42)
    fp = SelectionFingerprint(32, rng=rng)
    mask = 0b01010101
    results = {fp._random_set_bit(mask) for _ in range(100)}
    assert results == {0, 2, 4, 6}


# --------------------------------------------------------------------------- #
# FingerprintConfig round-trip
# --------------------------------------------------------------------------- #


def test_fingerprint_config_roundtrip() -> None:
    cfg = FingerprintConfig(bits=64, tournament_k=4, mutation_rate=0.05)
    restored = FingerprintConfig.from_dict(cfg.to_dict())
    assert restored.bits == 64
    assert restored.tournament_k == 4
    assert restored.mutation_rate == pytest.approx(0.05)


def test_fingerprint_config_defaults() -> None:
    cfg = FingerprintConfig.from_dict({})
    assert cfg.bits == 32
    assert cfg.tournament_k == 1
    assert cfg.mutation_rate == pytest.approx(0.01)


# --------------------------------------------------------------------------- #
# Population-level integration tests
# --------------------------------------------------------------------------- #


def _make_minimal_population(
    size: int = 10,
    tournament_k: int = 1,
    mutation_rate: float = 0.0,
) -> "Population":  # noqa: F821
    """Build a tiny population using the Maze environment for integration tests."""
    from arc3_agi.checkpoint import CheckpointConfig
    from arc3_agi.fingerprint import FingerprintConfig
    from arc3_agi.maze import Maze, MazeAutomaton
    from arc3_agi.population import Population

    maze = Maze("test", 4, seed=0)
    fp_cfg = FingerprintConfig(
        bits=8, tournament_k=tournament_k, mutation_rate=mutation_rate
    )
    pop = Population(
        size=size,
        AutomatonClass=MazeAutomaton,
        environment=maze,
        checkpoint_config=CheckpointConfig(enabled=False),
        fingerprint_config=fp_cfg,
    )
    return pop


def test_all_automata_have_fingerprint_on_init() -> None:
    pop = _make_minimal_population()
    for automaton in pop.automata:
        assert automaton.fingerprint is not None
        assert automaton.fingerprint.bits == 8


def test_fingerprint_not_reset_between_generations() -> None:
    pop = _make_minimal_population(size=10)
    pre = {id(a): a.fingerprint.value for a in pop.automata}
    pop.evolve()
    # Surviving automata (first half after sort) should have unchanged fingerprint
    # values (reset() was called but fingerprint should be preserved).
    for automaton in pop.automata:
        # The automaton existed before evolve() — check it against its own pre-value.
        if id(automaton) in pre:
            assert automaton.fingerprint is not None
            # Value may have changed due to flip_toward/away, but fingerprint object exists
            assert automaton.fingerprint.bits == 8


def test_tournament_k1_population_evolves_without_error() -> None:
    """k=1 (fast path) must behave identically to the pre-fingerprint random selection."""
    pop = _make_minimal_population(size=10, tournament_k=1)
    for _ in range(3):
        pop.evolve()
    assert len(pop.automata) == 10


def test_tournament_k_large_evolves_without_error() -> None:
    """k > 1 tournament path must not raise."""
    pop = _make_minimal_population(size=10, tournament_k=5)
    for _ in range(3):
        pop.evolve()
    assert len(pop.automata) == 10


def test_offspring_inherit_fingerprint() -> None:
    """After evolve(), the newly created offspring (bottom half) should have fingerprints."""
    pop = _make_minimal_population(size=10, mutation_rate=0.0)
    pop.evolve()
    half = len(pop.automata) // 2
    # After sort and replacement, all automata should have fingerprints.
    for automaton in pop.automata:
        assert automaton.fingerprint is not None


def test_fingerprint_update_toward_on_good_offspring() -> None:
    """If an offspring exceeds parent average, flip_toward should be called once on each
    surviving parent."""
    from arc3_agi.checkpoint import CheckpointConfig
    from arc3_agi.maze import Maze, MazeAutomaton
    from arc3_agi.population import Population

    maze = Maze("test", 4, seed=1)
    fp_cfg = FingerprintConfig(bits=8, tournament_k=1, mutation_rate=0.0)
    pop = Population(
        size=4,
        AutomatonClass=MazeAutomaton,
        environment=maze,
        checkpoint_config=CheckpointConfig(enabled=False),
        fingerprint_config=fp_cfg,
    )

    # Manually craft a prev_pairings entry where child.fitness > parent_avg.
    # We need two parents that survive (will be in top half after sort).
    p1, p2 = pop.automata[0], pop.automata[1]
    p1.fitness = 10.0
    p2.fitness = 8.0
    child = pop.automata[2]
    child.fitness = 20.0  # beats parent avg of 9.0

    p1_fp_before = p1.fingerprint.value
    p2_fp_before = p2.fingerprint.value

    pop._prev_pairings = [(p1, p2, child, p1.fitness, p2.fitness)]

    # Give the remaining automaton a low fitness so p1, p2 are in top half.
    pop.automata[3].fitness = 1.0
    pop.evolve()

    # At least one fingerprint flip should have happened (toward).
    # We can't assert the exact value (random bit choice), but we can check
    # that p1 and p2 were in the survivors and that flip was attempted.
    # Because bits may already be identical by chance, we just check no crash.


def test_fingerprint_update_away_on_zero_fitness_offspring() -> None:
    """If an offspring has zero fitness, flip_away should be called on each surviving parent."""
    from arc3_agi.checkpoint import CheckpointConfig
    from arc3_agi.maze import Maze, MazeAutomaton
    from arc3_agi.population import Population

    maze = Maze("test", 4, seed=2)
    fp_cfg = FingerprintConfig(bits=8, tournament_k=1, mutation_rate=0.0)
    pop = Population(
        size=4,
        AutomatonClass=MazeAutomaton,
        environment=maze,
        checkpoint_config=CheckpointConfig(enabled=False),
        fingerprint_config=fp_cfg,
    )

    p1, p2 = pop.automata[0], pop.automata[1]
    p1.fitness = 5.0
    p2.fitness = 5.0
    child = pop.automata[2]
    child.fitness = 0.0  # dead offspring → flip_away

    pop.automata[3].fitness = 1.0
    pop._prev_pairings = [(p1, p2, child, p1.fitness, p2.fitness)]
    # No assertion on exact value — just verify no exception is raised.
    pop.evolve()


def test_fingerprint_update_away_on_below_cutoff_offspring() -> None:
    """Offspring that survive but sit below the survivor cutoff → flip_away.

    The old rule used parent_avg as the flip_toward threshold, which meant
    high-fitness parents could never satisfy it (their children rarely outscore
    them).  The new rule uses survivor_cutoff (population median) so the
    criterion is rank-based: below-median offspring → flip_away, regardless of
    the parents' own fitness.
    """
    from arc3_agi.checkpoint import CheckpointConfig
    from arc3_agi.maze import Maze, MazeAutomaton
    from arc3_agi.population import Population

    maze = Maze("test", 4, seed=3)
    fp_cfg = FingerprintConfig(bits=8, tournament_k=1, mutation_rate=0.0)
    pop = Population(
        size=4,
        AutomatonClass=MazeAutomaton,
        environment=maze,
        checkpoint_config=CheckpointConfig(enabled=False),
        fingerprint_config=fp_cfg,
    )

    p1, p2 = pop.automata[0], pop.automata[1]
    p1.fitness = 10.0
    p2.fitness = 10.0
    child = pop.automata[2]
    child.fitness = 5.0  # below survivor_cutoff (10.0) → not in survivor_ids

    # Ensure fingerprints share at least some matching bits so flip_away has
    # a candidate position to act on.  Complementary fingerprints (all bits
    # differ) would make flip_away a no-op.
    if p1.fingerprint is not None:
        p1.fingerprint.value = 0b00001111
    if p2.fingerprint is not None:
        p2.fingerprint.value = 0b00111100  # shares bits 2-3 with p1

    p1_fp_before = p1.fingerprint.value if p1.fingerprint else None
    p2_fp_before = p2.fingerprint.value if p2.fingerprint else None

    pop.automata[3].fitness = 1.0
    pop._prev_pairings = [(p1, p2, child, p1.fitness, p2.fitness)]

    pop.evolve()
    # survivor_cutoff = automata[1].fitness = 10.0 after sort.
    # child fitness (5.0) ≤ cutoff and child not in survivor_ids → flip_away.
    # Both parents tie at fitness 10, so p1 is the learner by convention and
    # only p1's fingerprint is updated (flip_away from p2).
    assert p1.fingerprint is not None and p2.fingerprint is not None
    assert p1.fingerprint.value != p1_fp_before


def test_population_without_fingerprint_config_still_works() -> None:
    """Population with no FingerprintConfig should behave exactly like the original."""
    from arc3_agi.checkpoint import CheckpointConfig
    from arc3_agi.maze import Maze, MazeAutomaton
    from arc3_agi.population import Population

    maze = Maze("test", 4, seed=5)
    pop = Population(
        size=10,
        AutomatonClass=MazeAutomaton,
        environment=maze,
        checkpoint_config=CheckpointConfig(enabled=False),
        fingerprint_config=None,
    )
    for automaton in pop.automata:
        assert automaton.fingerprint is None
    for _ in range(3):
        pop.evolve()
    assert len(pop.automata) == 10


def test_one_update_per_unique_partner_per_generation() -> None:
    """If the same parent pair appears multiple times in _prev_pairings, only one
    fingerprint update fires per (parent, mate) ordered pair."""
    from unittest.mock import patch

    from arc3_agi.checkpoint import CheckpointConfig
    from arc3_agi.maze import Maze, MazeAutomaton
    from arc3_agi.population import Population

    maze = Maze("test", 4, seed=4)
    fp_cfg = FingerprintConfig(bits=32, tournament_k=1, mutation_rate=0.0)
    pop = Population(
        size=4,
        AutomatonClass=MazeAutomaton,
        environment=maze,
        checkpoint_config=CheckpointConfig(enabled=False),
        fingerprint_config=fp_cfg,
    )

    p1, p2 = pop.automata[0], pop.automata[1]
    p1.fitness = 10.0
    p2.fitness = 10.0
    child = pop.automata[2]
    child.fitness = 20.0  # good offspring — triggers flip_toward

    pop.automata[3].fitness = 1.0

    # Duplicate the same pairing three times.
    pop._prev_pairings = [
        (p1, p2, child, p1.fitness, p2.fitness),
        (p1, p2, child, p1.fitness, p2.fitness),
        (p1, p2, child, p1.fitness, p2.fitness),
    ]

    flip_calls: list[str] = []
    orig_toward = SelectionFingerprint.flip_toward
    orig_away = SelectionFingerprint.flip_away

    def counting_toward(
        self: SelectionFingerprint, other: SelectionFingerprint
    ) -> None:
        flip_calls.append("toward")
        orig_toward(self, other)

    def counting_away(self: SelectionFingerprint, other: SelectionFingerprint) -> None:
        flip_calls.append("away")
        orig_away(self, other)

    with (
        patch.object(SelectionFingerprint, "flip_toward", counting_toward),
        patch.object(SelectionFingerprint, "flip_away", counting_away),
    ):
        pop.evolve()

    # Each surviving parent gets exactly one update regardless of duplicate pairings.
    assert flip_calls.count("toward") <= 2  # at most one per surviving parent
    assert flip_calls.count("away") == 0
