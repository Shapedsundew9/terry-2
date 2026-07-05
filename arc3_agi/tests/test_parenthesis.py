"""Tests for the Parenthesis environment and its automaton."""

import random

import pytest

from arc3_agi.automaton import ActionStatus
from arc3_agi.genetic_code import GeneticCodeDict
from arc3_agi.parethesis import (
    Parenthesis,
    ParenthesisAction,
    ParenthesisAutomaton,
    ParenthesisToken,
)

OPEN = ParenthesisToken.OPEN.value  # 1
CLOSE = ParenthesisToken.CLOSE.value  # 2


@pytest.fixture
def env() -> Parenthesis:
    random.seed(1234)
    return Parenthesis(length=6, valid=True)


# --------------------------------------------------------------------------- #
# Sequence validation logic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "sequence, expected",
    [
        ([], True),
        ([OPEN, CLOSE], True),
        ([OPEN, OPEN, CLOSE, CLOSE], True),
        ([OPEN, CLOSE, OPEN, CLOSE], True),
        ([CLOSE, OPEN], False),  # negative balance immediately
        ([OPEN, CLOSE, CLOSE], False),  # dips negative
        ([OPEN, OPEN, CLOSE], False),  # ends unbalanced
    ],
)
def test_is_valid(env: Parenthesis, sequence: list[int], expected: bool) -> None:
    assert env.is_valid(sequence) is expected


def test_validation_state_marks_each_position(env: Parenthesis) -> None:
    assert env.validation_state([OPEN, CLOSE]) == [
        ParenthesisAction.CONSISTENT,
        ParenthesisAction.VALID,
    ]
    assert env.validation_state([CLOSE]) == [ParenthesisAction.INVALID]
    assert env.validation_state([OPEN, OPEN, CLOSE, CLOSE]) == [
        ParenthesisAction.CONSISTENT,
        ParenthesisAction.CONSISTENT,
        ParenthesisAction.CONSISTENT,
        ParenthesisAction.VALID,
    ]


def test_is_end(env: Parenthesis) -> None:
    length = len(env._array)
    assert env.is_end(length) is True
    assert env.is_end(length - 1) is False


def test_generate_valid_sequence_is_valid() -> None:
    random.seed(0)
    env = Parenthesis(length=8)
    assert env.is_valid(env.generate_valid_sequence(8)) is True


def test_generate_invalid_sequence_is_invalid() -> None:
    random.seed(0)
    env = Parenthesis(length=8)
    assert env.is_valid(env.generate_invalid_sequence(8)) is False


# --------------------------------------------------------------------------- #
# ParenthesisAutomaton
# --------------------------------------------------------------------------- #
def _fixed_env(sequence: list[int]) -> Parenthesis:
    env = Parenthesis(length=2)
    env._array = sequence
    env._validation_state = env.validation_state(sequence)
    return env


def test_attempt_action_rewards_correct_and_advances() -> None:
    env = _fixed_env([OPEN, CLOSE])  # validation: CONSISTENT(0), VALID(1)
    auto = ParenthesisAutomaton(environment=env)
    assert auto.coords == [0]

    status = auto.attempt_action(ParenthesisAction.CONSISTENT.value)
    assert status == ActionStatus.SUCCEEDED
    assert auto.fitness == 1.0
    assert auto.coords == [1]

    status = auto.attempt_action(ParenthesisAction.VALID.value)
    assert status == ActionStatus.FAILED  # reached end of sequence
    assert auto.fitness == 2.0


def test_attempt_action_penalises_wrong_and_invalid_holds_position() -> None:
    env = _fixed_env([OPEN, CLOSE])  # position 0 expects CONSISTENT
    auto = ParenthesisAutomaton(environment=env)

    status = auto.attempt_action(ParenthesisAction.INVALID.value)
    assert status == ActionStatus.SUCCEEDED
    assert auto.fitness == -1.0
    assert auto.coords == [0]  # INVALID does not advance


def test_tick_reports_done_at_end_of_sequence() -> None:
    env = _fixed_env([OPEN, CLOSE])
    # Force a CONSISTENT (0) response for every encountered input code so the
    # automaton always advances; with internal state staying 0, the input code
    # is just the current token (OPEN=1, then CLOSE=2).
    code = GeneticCodeDict({1: 0, 2: 0}, resp_bits=10)
    auto = ParenthesisAutomaton(environment=env, genetic_code=code)
    auto.internal_state = 0  # ensure internal state is 0 to match code

    assert auto.tick() == ParenthesisAction.CONSISTENT.value  # position 0
    assert auto.coords == [1]

    assert auto.tick() == ParenthesisAction.DONE.value  # advances off the end
    assert auto.last_action == ParenthesisAction.DONE.value


def test_dict_code_drives_tick_within_response_mask() -> None:
    env = _fixed_env([OPEN, CLOSE, OPEN, CLOSE])
    code = GeneticCodeDict({}, seed=11, resp_bits=10)
    auto = ParenthesisAutomaton(environment=env, genetic_code=code)
    for _ in range(10):
        result = auto.tick()
        assert 0 <= result <= ParenthesisAction.DONE.value
        if result == ParenthesisAction.DONE.value:
            break
