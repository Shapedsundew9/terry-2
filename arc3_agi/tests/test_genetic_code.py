"""Tests for the GeneticCode implementations.

The cross-implementation tests assert that ``GeneticCodeDict``,
``GeneticCodeList`` and ``GeneticCodeGraph`` all honour the same contract so the
new graph-based code interacts with the rest of the system identically to the
existing table-based codes. Implementation-specific behaviour (the logic graph
in particular) is covered by the dedicated sections below.
"""

import random

import pytest

from arc3_agi.genetic_code import (
    GeneticCode,
    GeneticCodeDict,
    GeneticCodeGraph,
    GeneticCodeList,
)

RESP_BITS = 6
RESP_LIMIT = 1 << RESP_BITS
SAMPLE_KEYS = list(range(64))


def make_code(impl: str, seed: int) -> GeneticCode:
    """Build a populated code of the requested implementation."""
    if impl == "dict":
        return GeneticCodeDict({}, seed=seed, resp_bits=RESP_BITS)
    if impl == "list":
        rng = random.Random(seed)
        values = [rng.getrandbits(RESP_BITS) for _ in range(len(SAMPLE_KEYS))]
        return GeneticCodeList(values, seed=seed, resp_bits=RESP_BITS)
    if impl == "graph":
        return GeneticCodeGraph.random(
            input_bits=6, resp_bits=RESP_BITS, num_nodes=20, seed=seed
        )
    raise ValueError(impl)


@pytest.fixture(params=["dict", "list", "graph"])
def impl(request) -> str:
    return request.param


# --------------------------------------------------------------------------- #
# Shared contract across all implementations
# --------------------------------------------------------------------------- #
def test_output_within_resp_bits(impl: str) -> None:
    code = make_code(impl, seed=1)
    for key in SAMPLE_KEYS:
        value = code[key]
        assert 0 <= value < RESP_LIMIT


def test_lookup_is_deterministic(impl: str) -> None:
    code = make_code(impl, seed=2)
    for key in SAMPLE_KEYS:
        assert code[key] == code[key]


def test_same_seed_is_reproducible(impl: str) -> None:
    a = make_code(impl, seed=99)
    b = make_code(impl, seed=99)
    assert [a[k] for k in SAMPLE_KEYS] == [b[k] for k in SAMPLE_KEYS]


def test_crossover_preserves_type_and_width(impl: str) -> None:
    parent1 = make_code(impl, seed=10)
    parent2 = make_code(impl, seed=20)
    child = parent1.crossover(parent2, mutation_rate=0.05)
    assert type(child) is type(parent1)
    assert child.resp_bits == RESP_BITS
    for key in SAMPLE_KEYS:
        assert 0 <= child[key] < RESP_LIMIT


# --------------------------------------------------------------------------- #
# GeneticCodeDict specifics
# --------------------------------------------------------------------------- #
def test_dict_lazily_generates_and_stores_unknown_keys() -> None:
    code = GeneticCodeDict({}, seed=5, resp_bits=4)
    assert 123 not in code
    value = code[123]
    assert 123 in code
    assert code[123] == value  # stored, so stable


# --------------------------------------------------------------------------- #
# GeneticCodeList specifics
# --------------------------------------------------------------------------- #
def test_list_indexes_directly() -> None:
    code = GeneticCodeList([3, 1, 2], resp_bits=2)
    assert code[0] == 3
    assert code[1] == 1
    assert code[2] == 2
    assert len(code) == 3


# --------------------------------------------------------------------------- #
# GeneticCodeGraph specifics
# --------------------------------------------------------------------------- #
def test_graph_ignores_bits_above_input_width() -> None:
    code = GeneticCodeGraph.random(input_bits=8, resp_bits=5, num_nodes=16, seed=4)
    for key in range(256):
        assert code[key] == code[key + (1 << 8)]


def test_graph_is_not_constant() -> None:
    code = GeneticCodeGraph.random(input_bits=8, resp_bits=5, num_nodes=24, seed=3)
    outputs = {code[k] for k in range(256)}
    assert len(outputs) > 1


def test_graph_genome_round_trips() -> None:
    code = GeneticCodeGraph.random(input_bits=8, resp_bits=5, num_nodes=16, seed=6)
    clone = GeneticCodeGraph(tuple(code._nodes), resp_bits=5, input_bits=8)
    for key in range(256):
        assert code[key] == clone[key]


def test_graph_zero_mutation_self_crossover_is_identity() -> None:
    code = GeneticCodeGraph.random(input_bits=8, resp_bits=5, num_nodes=16, seed=7)
    child = code.crossover(code, mutation_rate=0.0)
    for key in range(256):
        assert child[key] == code[key]


def test_graph_mutation_changes_some_output() -> None:
    code = GeneticCodeGraph.random(input_bits=8, resp_bits=5, num_nodes=24, seed=8)
    mutated = code.crossover(code, mutation_rate=0.9)
    assert any(mutated[k] != code[k] for k in range(256))


def test_graph_crossover_rejects_other_implementations() -> None:
    graph = GeneticCodeGraph.random(input_bits=8, resp_bits=5, num_nodes=16, seed=9)
    with pytest.raises(AssertionError):
        graph.crossover(GeneticCodeDict({}, resp_bits=5))


def test_graph_crossover_requires_matching_shape() -> None:
    a = GeneticCodeGraph.random(input_bits=8, resp_bits=5, num_nodes=16, seed=1)
    b = GeneticCodeGraph.random(input_bits=8, resp_bits=5, num_nodes=20, seed=1)
    with pytest.raises(AssertionError):
        a.crossover(b)


def test_graph_mutators_are_not_supported() -> None:
    code = GeneticCodeGraph.random(input_bits=8, resp_bits=5, num_nodes=16, seed=1)
    with pytest.raises(NotImplementedError):
        code[0] = 1
    with pytest.raises(NotImplementedError):
        del code[0]
    with pytest.raises(NotImplementedError):
        iter(code)
    with pytest.raises(NotImplementedError):
        len(code)
