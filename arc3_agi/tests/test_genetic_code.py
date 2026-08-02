"""Tests for the GeneticCode implementations.

The cross-implementation tests assert that ``GeneticCodeDict`` and
``GeneticCodeList`` both honour the same contract. Implementation-specific
behaviour is covered by the dedicated sections below.
"""

import random

import numpy as np
import pytest

from arc3_agi.genetic_code import (
    GeneticCode,
    GeneticCodeDict,
    GeneticCodeList,
    GeneticCodeTsetlin,
    _mutation_mask_depth,
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
    raise ValueError(impl)


@pytest.fixture(params=["dict", "list"])
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
# GeneticCodeTsetlin specifics
# --------------------------------------------------------------------------- #
def _make_tsetlin(
    w_pos: list[list[int]],
    w_neg: list[list[int]],
    *,
    seed: int,
    input_bits: int = 8,
) -> GeneticCodeTsetlin:
    pos = np.asarray(w_pos, dtype=np.uint64)
    neg = np.asarray(w_neg, dtype=np.uint64)
    return GeneticCodeTsetlin(
        code=(pos, neg),
        seed=seed,
        resp_bits=pos.shape[0],
        num_clauses=pos.shape[1],
        input_bits=input_bits,
    )


def test_tsetlin_defaults_to_four_clauses_with_strict_majority() -> None:
    code = GeneticCodeTsetlin(seed=1, resp_bits=2, input_bits=8)

    assert code.num_clauses == 4
    assert code.threshold == 3


@pytest.mark.parametrize(
    ("mutation_rate", "depth"),
    [(1.0, 0), (0.5, 1), (0.01, 7), (0.001, 10)],
)
def test_tsetlin_mutation_rate_maps_to_nearest_power_of_two(
    mutation_rate: float, depth: int
) -> None:
    assert _mutation_mask_depth(mutation_rate) == depth


def test_tsetlin_crossover_is_reproducible_and_inherits_whole_clauses() -> None:
    parent1 = _make_tsetlin(
        [[1, 2, 4], [8, 16, 32]],
        [[64, 128, 3], [5, 6, 9]],
        seed=41,
    )
    parent1_copy = _make_tsetlin(
        [[1, 2, 4], [8, 16, 32]],
        [[64, 128, 3], [5, 6, 9]],
        seed=41,
    )
    parent2 = _make_tsetlin(
        [[10, 20, 40], [80, 7, 11]],
        [[1, 1, 1], [1, 16, 32]],
        seed=99,
    )

    child = parent1.crossover(parent2, mutation_rate=0.0)
    duplicate = parent1_copy.crossover(parent2, mutation_rate=0.0)

    assert child.num_clauses == parent1.num_clauses
    assert child.threshold == parent1.threshold
    assert np.array_equal(child._w_pos, duplicate._w_pos)
    assert np.array_equal(child._w_neg, duplicate._w_neg)
    for response_bit in range(child.resp_bits):
        for clause_index in range(child.num_clauses):
            child_pair = (
                int(child._w_pos[response_bit, clause_index]),
                int(child._w_neg[response_bit, clause_index]),
            )
            parent1_pair = (
                int(parent1._w_pos[response_bit, clause_index]),
                int(parent1._w_neg[response_bit, clause_index]),
            )
            parent2_pair = (
                int(parent2._w_pos[response_bit, clause_index]),
                int(parent2._w_neg[response_bit, clause_index]),
            )
            assert child_pair in (parent1_pair, parent2_pair)


def test_tsetlin_forced_mutation_changes_every_valid_literal_per_clause() -> None:
    parent1 = _make_tsetlin(
        [[0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0]],
        seed=7,
        input_bits=8,
    )
    parent2 = _make_tsetlin(
        [[0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0]],
        seed=8,
        input_bits=8,
    )

    child = parent1.crossover(parent2, mutation_rate=1.0)

    assert child.num_clauses == parent1.num_clauses
    assert child.threshold == 2
    assert np.all((child._w_pos & child._w_neg) == 0)
    literal_masks = child._w_pos | child._w_neg
    assert np.all(literal_masks == np.uint64(0xFF))


def test_tsetlin_mutation_preserves_single_clause() -> None:
    parent1 = _make_tsetlin([[0]], [[0]], seed=17, input_bits=4)
    parent2 = _make_tsetlin([[0]], [[0]], seed=18, input_bits=4)

    child = parent1.crossover(parent2, mutation_rate=1.0)

    assert child.num_clauses == 1
    assert child.threshold == 1


def test_tsetlin_rejects_contradictory_clause_masks() -> None:
    with pytest.raises(ValueError, match="contradictory"):
        _make_tsetlin([[0b001]], [[0b001]], seed=1)


@pytest.mark.parametrize("mutation_rate", [-0.01, 1.01, float("nan")])
def test_tsetlin_crossover_rejects_invalid_mutation_rate(
    mutation_rate: float,
) -> None:
    parent = GeneticCodeTsetlin(seed=1, resp_bits=2, num_clauses=2, input_bits=8)

    with pytest.raises(ValueError, match="mutation_rate"):
        parent.crossover(parent, mutation_rate=mutation_rate)


def test_tsetlin_crossover_rejects_incompatible_parent() -> None:
    parent = GeneticCodeTsetlin(seed=1, resp_bits=2, num_clauses=2, input_bits=8)

    with pytest.raises(TypeError, match="GeneticCodeTsetlin"):
        parent.crossover(GeneticCodeDict({}, seed=2, resp_bits=2))
    with pytest.raises(ValueError, match="response bits"):
        parent.crossover(
            GeneticCodeTsetlin(seed=2, resp_bits=3, num_clauses=2, input_bits=8)
        )
    with pytest.raises(ValueError, match="input bits"):
        parent.crossover(
            GeneticCodeTsetlin(seed=2, resp_bits=2, num_clauses=2, input_bits=7)
        )
    with pytest.raises(ValueError, match="clause count"):
        parent.crossover(
            GeneticCodeTsetlin(seed=2, resp_bits=2, num_clauses=3, input_bits=8)
        )
