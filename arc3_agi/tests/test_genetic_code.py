"""Tests for the GeneticCode implementations.

The cross-implementation tests assert that ``GeneticCodeDict`` and
``GeneticCodeList`` both honour the same contract. Implementation-specific
behaviour is covered by the dedicated sections below.
"""

import random

import pytest

from arc3_agi.genetic_code import (
    GeneticCode,
    GeneticCodeDict,
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
