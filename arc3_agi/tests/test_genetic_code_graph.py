"""Tests for GeneticCodeGraph — the empirical behavioral-graph crossover implementation.

Coverage:
  * __getitem__ builds the transition graph from consecutive calls.
  * reset_trajectory() clears only the last-key cursor (not the graph).
  * Graph is NOT built across episode boundaries (cursor cleared on reset).
  * crossover() returns a GeneticCodeGraph with correct resp_bits / edge_threshold.
  * Child entries stay within resp_bits.
  * BFS pull: entries connected via edges >= threshold ARE pulled into child.
  * BFS pull: entries connected via edges < threshold are NOT pulled.
  * BFS pull: child mappings are never overwritten (first-claim wins).
  * Tie-break: secondary's value used when it has a strictly higher edge count.
  * Mutation is applied (geometric sampling).
  * Serialization round-trip: to_dict / from_dict restores code but NOT graph.
  * Checkpoint factory dispatches to GeneticCodeGraph correctly.
  * AutomatonBase.reset() calls reset_trajectory() on the genetic code.
"""

import pytest

from arc3_agi.genetic_code import GeneticCodeDict, GeneticCodeGraph

RESP_BITS = 4
RESP_LIMIT = 1 << RESP_BITS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gc(code: dict[int, int], seed: int = 0, threshold: int = 3) -> GeneticCodeGraph:
    return GeneticCodeGraph(
        code, seed=seed, resp_bits=RESP_BITS, edge_threshold=threshold
    )


def _populated(seed: int = 42, n: int = 16) -> GeneticCodeGraph:
    """Build a code with n entries (keys 0..n-1) with random values."""
    import random

    rng = random.Random(seed)
    code = {k: rng.randrange(RESP_LIMIT) for k in range(n)}
    return _gc(code, seed=seed)


def _simulate_ticks(gc: GeneticCodeGraph, sequence: list[int]) -> None:
    """Drive __getitem__ with a fixed key sequence to populate the graph."""
    for key in sequence:
        gc[key]


# ---------------------------------------------------------------------------
# Runtime graph building
# ---------------------------------------------------------------------------


def test_graph_built_from_consecutive_getitem() -> None:
    gc = _gc({0: 1, 1: 2, 2: 3})
    _simulate_ticks(gc, [0, 1, 2])
    # Edge 0→1 and 1→2 should both have count 1
    assert gc._graph.get(0, {}).get(1, 0) == 1
    assert gc._graph.get(1, {}).get(2, 0) == 1
    # No edge 0→2 (non-consecutive)
    assert gc._graph.get(0, {}).get(2, 0) == 0


def test_graph_counts_accumulate() -> None:
    gc = _gc({0: 10, 1: 20})
    # Simulate the same 0→1 transition three times
    for _ in range(3):
        gc.reset_trajectory()
        _simulate_ticks(gc, [0, 1])
    assert gc._graph[0][1] == 3


def test_reset_trajectory_clears_cursor_not_graph() -> None:
    gc = _gc({0: 1, 1: 2})
    _simulate_ticks(gc, [0, 1])
    assert gc._graph[0][1] == 1  # graph populated
    gc.reset_trajectory()
    assert gc._last_key is None  # cursor cleared
    assert gc._graph[0][1] == 1  # graph preserved


def test_no_edge_recorded_at_episode_start() -> None:
    """The very first key after reset_trajectory must not create a back-edge."""
    gc = _gc({5: 9, 9: 1})
    gc.reset_trajectory()
    gc[5]  # first query — no previous key, so no edge should be recorded
    assert not gc._graph  # graph should still be empty


def test_no_cross_episode_edges() -> None:
    """Edges must not bridge across episode boundaries (cursor cleared by reset)."""
    gc = _gc({0: 1, 7: 2})
    _simulate_ticks(gc, [0])  # episode 1 ends on key 0
    gc.reset_trajectory()
    _simulate_ticks(gc, [7])  # episode 2 starts on key 7
    # There should be no edge 0→7
    assert gc._graph.get(0, {}).get(7, 0) == 0


# ---------------------------------------------------------------------------
# crossover() — structural checks
# ---------------------------------------------------------------------------


def test_crossover_returns_genetic_code_graph() -> None:
    a, b = _populated(1), _populated(2)
    child = a.crossover(b, mutation_rate=0.0)
    assert isinstance(child, GeneticCodeGraph)


def test_crossover_preserves_resp_bits() -> None:
    a, b = _populated(3), _populated(4)
    child = a.crossover(b, mutation_rate=0.0)
    assert child.resp_bits == RESP_BITS


def test_crossover_preserves_edge_threshold() -> None:
    a = GeneticCodeGraph({}, resp_bits=RESP_BITS, edge_threshold=7)
    b = GeneticCodeGraph({}, resp_bits=RESP_BITS, edge_threshold=7)
    child = a.crossover(b, mutation_rate=0.0)
    assert child.edge_threshold == 7


def test_crossover_child_values_within_resp_bits() -> None:
    a, b = _populated(5), _populated(6)
    for _ in range(5):
        child = a.crossover(b, mutation_rate=0.0)
        for v in child._code.values():
            assert 0 <= v < RESP_LIMIT


def test_crossover_wrong_type_raises() -> None:
    a = _populated(7)
    bad = GeneticCodeDict({}, resp_bits=RESP_BITS)
    with pytest.raises(AssertionError):
        a.crossover(bad)


def test_crossover_child_graph_starts_empty() -> None:
    """Child's behavioral graph must be empty — fresh start, not inherited."""
    a, b = _populated(8), _populated(9)
    # Simulate some ticks to populate parents' graphs
    _simulate_ticks(a, list(range(16)) * 5)
    _simulate_ticks(b, list(range(16)) * 5)
    child = a.crossover(b, mutation_rate=0.0)
    assert child._graph == {}
    assert child._last_key is None


# ---------------------------------------------------------------------------
# BFS pull behaviour
# ---------------------------------------------------------------------------


def _make_chained_pair(threshold: int = 3) -> tuple[GeneticCodeGraph, GeneticCodeGraph]:
    """Create two parents where parent A has a chain 10→11→12 at high count."""
    a = _gc({10: 99, 11: 88, 12: 77, 13: 66}, threshold=threshold)
    b = _gc({10: 11, 11: 22, 12: 33, 13: 44}, threshold=threshold)

    # Build parent A's graph: 10→11 at count 5, 11→12 at count 5 (above threshold=3)
    # 12→13 at count 1 (below threshold)
    for _ in range(5):
        a.reset_trajectory()
        _simulate_ticks(a, [10, 11, 12])
    a.reset_trajectory()
    _simulate_ticks(a, [12, 13])  # 12→13: count=1 only

    return a, b


def test_bfs_pulls_above_threshold() -> None:
    """Keys reachable via edges >= threshold should be pulled in alongside primary key."""
    a, b = _make_chained_pair(threshold=3)
    # Force key 10 to be selected from parent A first (seed 0, first shuffle)
    # by doing multiple crossovers and checking at least one pulls 11.
    pulled_11 = False
    for seed in range(20):
        a_copy = _gc(dict(a._code), seed=seed, threshold=3)
        a_copy._graph = {k: dict(v) for k, v in a._graph.items()}
        b_copy = _gc(dict(b._code), seed=seed, threshold=3)

        child = a_copy.crossover(b_copy, mutation_rate=0.0)
        # If 10 is in child from A, 11 must also be in child (edge count=5 >= 3)
        if child._code.get(10) == a._code[10] and 11 in child._code:
            pulled_11 = True
            break
    assert pulled_11, "BFS pull should bring key 11 in when selected from parent A"


def test_bfs_does_not_pull_below_threshold() -> None:
    """Key 13 (edge count=1 < threshold=3) must not enter the child via BFS from key 12."""
    a, b = _make_chained_pair(threshold=3)

    # Run many crossovers and check the invariant directly via _bfs_pull:
    # if we seed child_code with key 12 already in it (from A), BFS from 12
    # must NOT add key 13 because the 12→13 edge has count=1 < threshold=3.
    child_code: dict[int, int] = {12: a._code[12]}
    a._bfs_pull(12, a, b, child_code, threshold=3)
    assert (
        13 not in child_code
    ), "Key 13 should not be pulled by BFS; its edge count (1) is below threshold (3)"


def test_bfs_does_not_overwrite_existing_child_entries() -> None:
    """A key already claimed by top-level selection must not be overwritten by BFS."""
    # Build parents where key 5 is reachable from key 3 in A's graph.
    a = _gc({3: 10, 5: 20}, threshold=2)
    b = _gc({3: 30, 5: 40}, threshold=2)

    # Build A's graph: 3→5 at count 5
    for _ in range(5):
        a.reset_trajectory()
        _simulate_ticks(a, [3, 5])

    # Force key 5 to be in child_code before BFS from 3 runs, by simulating
    # what would happen with a pre-seeded child. We test _bfs_pull directly.
    child_code: dict[int, int] = {5: 999}  # sentinel value
    a._bfs_pull(3, a, b, child_code, threshold=2)
    # Key 5 should NOT have been overwritten by BFS
    assert child_code[5] == 999


# ---------------------------------------------------------------------------
# Tie-break: secondary's value when count is strictly higher
# ---------------------------------------------------------------------------


def test_tiebreak_uses_secondary_value_when_higher_count() -> None:
    """When secondary has a strictly higher edge count, secondary's value is used."""
    sentinel_primary = 0b0001
    sentinel_secondary = 0b1110

    a = _gc({0: sentinel_primary, 1: 55}, threshold=1)
    b = _gc({0: sentinel_secondary, 1: 66}, threshold=1)

    # A's graph: 0→1 at count=2
    for _ in range(2):
        a.reset_trajectory()
        _simulate_ticks(a, [0, 1])

    # B's graph: 0→1 at count=5 (strictly higher)
    for _ in range(5):
        b.reset_trajectory()
        _simulate_ticks(b, [0, 1])

    # Direct BFS call: primary=A, secondary=B
    child_code: dict[int, int] = {0: sentinel_primary}  # key 0 already claimed from A
    a._bfs_pull(0, a, b, child_code, threshold=1)

    # Key 1 should be pulled, and since B has higher count, B's value (66) should be used
    assert 1 in child_code
    assert child_code[1] == b._code[1]  # 66, from secondary


def test_tiebreak_uses_primary_value_when_equal_or_lower() -> None:
    """When primary count >= secondary count, primary's value is kept."""
    a = _gc({0: 77, 1: 88}, threshold=1)
    b = _gc({0: 11, 1: 22}, threshold=1)

    # Equal counts: A=3, B=3
    for _ in range(3):
        a.reset_trajectory()
        _simulate_ticks(a, [0, 1])
        b.reset_trajectory()
        _simulate_ticks(b, [0, 1])

    child_code: dict[int, int] = {0: a._code[0]}
    a._bfs_pull(0, a, b, child_code, threshold=1)

    # Primary (A) count not exceeded, so primary's value (88) should be used for key 1
    assert child_code.get(1) == a._code[1]  # 88


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


def test_zero_mutation_no_bit_flips() -> None:
    """With mutation_rate=0 all child values come directly from parent code dicts."""
    a, b = _populated(10), _populated(11)
    parent_vals = set(a._code.values()) | set(b._code.values())
    child = a.crossover(b, mutation_rate=0.0)
    for v in child._code.values():
        assert v in parent_vals


def test_high_mutation_introduces_novelty() -> None:
    """mutation_rate=0.99 should produce values not in either parent."""
    a, b = _populated(12), _populated(13)
    parent_vals = set(a._code.values()) | set(b._code.values())
    child = a.crossover(b, mutation_rate=0.99)
    child_vals = set(child._code.values())
    # With 99% mutation virtually guaranteed to flip at least one entry
    assert not child_vals.issubset(parent_vals)


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_code() -> None:
    original = _populated(14)
    d = original.to_dict()
    arrays = original.to_arrays()

    assert d["type"] == "GeneticCodeGraph"
    assert d["edge_threshold"] == original.edge_threshold
    assert d["resp_bits"] == RESP_BITS

    restored = GeneticCodeGraph.from_dict(d, arrays)
    assert restored.resp_bits == RESP_BITS
    assert restored.edge_threshold == original.edge_threshold
    for k, v in original._code.items():
        assert restored._code[k] == v


def test_round_trip_graph_is_empty_after_restore() -> None:
    """The behavioral graph is ephemeral and must NOT be persisted."""
    gc = _populated(15)
    _simulate_ticks(gc, list(range(16)) * 3)
    assert gc._graph  # sanity: graph was built

    d = gc.to_dict()
    arrays = gc.to_arrays()
    restored = GeneticCodeGraph.from_dict(d, arrays)
    assert restored._graph == {}
    assert restored._last_key is None


def test_round_trip_via_checkpoint_factory() -> None:
    from arc3_agi.checkpoint import genetic_code_from_dict

    original = _populated(16)
    restored = genetic_code_from_dict(original.to_dict(), original.to_arrays())
    assert isinstance(restored, GeneticCodeGraph)
    for k, v in original._code.items():
        assert restored._code[k] == v


# ---------------------------------------------------------------------------
# AutomatonBase.reset() integration
# ---------------------------------------------------------------------------
