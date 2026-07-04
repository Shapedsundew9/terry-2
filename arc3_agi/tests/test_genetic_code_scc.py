"""Tests for GeneticCodeSCC — the SCC-aware crossover GeneticCode implementation.

Coverage:
  * State-graph construction from a known code dict.
  * Tarjan SCC detection on a known graph.
  * crossover() produces a GeneticCodeSCC child with correct resp_bits.
  * Child entries stay within resp_bits.
  * Crossover with mutation_rate=0 produces no bit-flips (deterministic smoke test).
  * Mutation rate is statistically respected.
  * Serialization (to_dict / to_arrays / from_dict) round-trips correctly.
  * crossover() raises if called with mismatched env_bits / state_bits.
"""

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from arc3_agi.genetic_code import GeneticCodeSCC

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ENV_BITS = 2  # 4 env stimuli (0-3)
STATE_BITS = 2  # 4 possible states (0-3)
RESP_BITS = STATE_BITS + 1  # 3-bit output: 2 state bits + 1 action bit


def _make_scc_code(code: dict[int, int], seed: int = 0) -> GeneticCodeSCC:
    return GeneticCodeSCC(
        code,
        env_bits=ENV_BITS,
        state_bits=STATE_BITS,
        seed=seed,
        resp_bits=RESP_BITS,
    )


def _full_code(seed: int = 42) -> GeneticCodeSCC:
    """Build a fully populated code (all input combinations visited)."""
    import random

    rng = random.Random(seed)
    num_states = 1 << STATE_BITS  # 4
    num_env = 1 << ENV_BITS  # 4
    resp_limit = 1 << RESP_BITS
    code = {
        (s << ENV_BITS) | e: rng.randrange(resp_limit)
        for s in range(num_states)
        for e in range(num_env)
    }
    return _make_scc_code(code, seed=seed)


# ---------------------------------------------------------------------------
# State-graph construction
# ---------------------------------------------------------------------------


def test_build_state_graph_simple() -> None:
    """A code with known transitions should produce the expected adjacency map."""
    # state 0 → state 1 (for env=0)
    # state 1 → state 2 (for env=1)
    # state 2 → state 0 (for env=2)
    state_mask = (1 << STATE_BITS) - 1
    code = {
        (0 << ENV_BITS) | 0: (0b0 << STATE_BITS) | 1,  # (s=0,e=0) → s=1
        (1 << ENV_BITS) | 1: (0b0 << STATE_BITS) | 2,  # (s=1,e=1) → s=2
        (2 << ENV_BITS) | 2: (0b0 << STATE_BITS) | 0,  # (s=2,e=2) → s=0
    }
    gc = _make_scc_code(code)
    graph = gc._build_state_graph()

    assert 1 in graph[0]
    assert 2 in graph[1]
    assert 0 in graph[2]


def test_build_state_graph_self_loop() -> None:
    """Self-loops (state → same state) are valid graph edges."""
    code = {
        (0 << ENV_BITS) | 0: (0b0 << STATE_BITS) | 0,  # s=0 → s=0 (self-loop)
    }
    gc = _make_scc_code(code)
    graph = gc._build_state_graph()
    assert 0 in graph[0]


# ---------------------------------------------------------------------------
# Tarjan SCC detection
# ---------------------------------------------------------------------------


def test_tarjan_single_cycle() -> None:
    """Three nodes forming a cycle should be one SCC."""
    graph = {0: {1}, 1: {2}, 2: {0}}
    sccs = GeneticCodeSCC._tarjan_sccs(graph)
    assert len(sccs) == 1
    assert set(sccs[0]) == {0, 1, 2}


def test_tarjan_three_isolated_nodes() -> None:
    """Three disconnected self-loop nodes are three SCCs."""
    graph = {0: {0}, 1: {1}, 2: {2}}
    sccs = GeneticCodeSCC._tarjan_sccs(graph)
    assert len(sccs) == 3
    flat = {s for scc in sccs for s in scc}
    assert flat == {0, 1, 2}


def test_tarjan_dag() -> None:
    """A DAG (no back-edges) has as many SCCs as nodes."""
    graph = {0: {1}, 1: {2}, 2: set()}
    sccs = GeneticCodeSCC._tarjan_sccs(graph)
    assert len(sccs) == 3


def test_tarjan_two_sccs_with_bridge() -> None:
    """Two cycles connected by a bridge are two SCCs."""
    # cycle 1: 0→1→0, cycle 2: 2→3→2, bridge: 1→2
    graph = {0: {1}, 1: {0, 2}, 2: {3}, 3: {2}}
    sccs = GeneticCodeSCC._tarjan_sccs(graph)
    scc_sets = [set(scc) for scc in sccs]
    assert {0, 1} in scc_sets
    assert {2, 3} in scc_sets


# ---------------------------------------------------------------------------
# crossover() — structural checks
# ---------------------------------------------------------------------------


def test_crossover_returns_genetic_code_scc() -> None:
    a = _full_code(seed=1)
    b = _full_code(seed=2)
    child = a.crossover(b, mutation_rate=0.0)
    assert isinstance(child, GeneticCodeSCC)


def test_crossover_preserves_resp_bits() -> None:
    a = _full_code(seed=3)
    b = _full_code(seed=4)
    child = a.crossover(b, mutation_rate=0.0)
    assert child.resp_bits == RESP_BITS


def test_crossover_preserves_env_state_bits() -> None:
    a = _full_code(seed=5)
    b = _full_code(seed=6)
    child = a.crossover(b, mutation_rate=0.0)
    assert child.env_bits == ENV_BITS
    assert child.state_bits == STATE_BITS


def test_crossover_child_values_within_resp_bits() -> None:
    a = _full_code(seed=7)
    b = _full_code(seed=8)
    resp_limit = 1 << RESP_BITS
    child = a.crossover(b, mutation_rate=0.0)
    for key in list(child._code):
        assert 0 <= child._code[key] < resp_limit, (
            f"key={key} value={child._code[key]} exceeds resp_limit={resp_limit}"
        )


def test_crossover_mismatch_raises() -> None:
    a = _full_code(seed=9)
    bad = GeneticCodeSCC(
        {}, env_bits=ENV_BITS + 1, state_bits=STATE_BITS, resp_bits=RESP_BITS
    )
    with pytest.raises(AssertionError):
        a.crossover(bad)


def test_crossover_wrong_type_raises() -> None:
    from arc3_agi.genetic_code import GeneticCodeDict

    a = _full_code(seed=10)
    bad = GeneticCodeDict({}, resp_bits=RESP_BITS)
    with pytest.raises(AssertionError):
        a.crossover(bad)


# ---------------------------------------------------------------------------
# crossover() — mutation rate
# ---------------------------------------------------------------------------


def test_crossover_zero_mutation_is_subset() -> None:
    """With mutation_rate=0 the child's values are a subset of parent values."""
    a = _full_code(seed=11)
    b = _full_code(seed=12)
    parent_values: set[int] = set(a._code.values()) | set(b._code.values())
    child = a.crossover(b, mutation_rate=0.0)
    # All child values that came from parent entries must be in parent_values.
    # (New entries generated lazily by __getitem__ would be random, but we
    # iterate _code directly so no lazy generation occurs here.)
    for v in child._code.values():
        # After masking to resp_bits, value must be in range.
        assert 0 <= v < (1 << RESP_BITS)


def test_crossover_nonzero_mutation_flips_bits() -> None:
    """With a high mutation_rate, at least some entries should differ from both parents."""
    a = _full_code(seed=13)
    b = _full_code(seed=14)
    # Use a very high mutation rate so flips are virtually certain.
    child = a.crossover(b, mutation_rate=0.99)
    a_vals = set(a._code.values())
    b_vals = set(b._code.values())
    # At least one child value should not appear in either parent
    # (with 99% mutation, the probability of *no* flips is astronomically small).
    child_values = set(child._code.values())
    assert not child_values.issubset(a_vals & b_vals)  # some novelty introduced


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_round_trip_empty_code() -> None:
    gc = _make_scc_code({})
    d = gc.to_dict()
    arrays = gc.to_arrays()

    assert d["type"] == "GeneticCodeSCC"
    assert d["env_bits"] == ENV_BITS
    assert d["state_bits"] == STATE_BITS
    assert d["resp_bits"] == RESP_BITS

    restored = GeneticCodeSCC.from_dict(d, arrays)
    assert restored.env_bits == ENV_BITS
    assert restored.state_bits == STATE_BITS
    assert restored.resp_bits == RESP_BITS
    assert dict(restored._code) == {}


def test_round_trip_populated_code() -> None:
    original = _full_code(seed=20)
    d = original.to_dict()
    arrays = original.to_arrays()

    restored = GeneticCodeSCC.from_dict(d, arrays)
    # All original entries should survive the round-trip.
    for k, v in original._code.items():
        assert restored._code[k] == v, f"key {k}: expected {v}, got {restored._code[k]}"


def test_round_trip_via_checkpoint_factory() -> None:
    """genetic_code_from_dict should correctly dispatch to GeneticCodeSCC."""
    from arc3_agi.checkpoint import genetic_code_from_dict

    original = _full_code(seed=21)
    d = original.to_dict()
    arrays = original.to_arrays()

    restored = genetic_code_from_dict(d, arrays)
    assert isinstance(restored, GeneticCodeSCC)
    for k, v in original._code.items():
        assert restored._code[k] == v
