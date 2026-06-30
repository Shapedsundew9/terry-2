from __future__ import annotations

from enum import IntEnum

from arc3_agi.environment import Environment
from arc3_agi.genetic_code import GeneticCodeDict


class ActionStatus(IntEnum):
    """Defines the possible action result statuses."""

    SUCCEEDED = 0  # All environment state was updated as attempted.
    FAILED = 1  # No environment state was updated.
    PARTIAL = 2  # Environment state updated but not as attempted.
    INVALID = 3  # The action was invalid. (Should never happen)


class AutomatonBase:
    """Base class for an automaton that interacts with an environment.

    The base class automaton contains no dynamic state so it does not learn or
    remember; it simply defines the interface.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes the automaton with a name and genetic code.

        Args:
            environment: The environment the automaton interacts with.
            genetic_code (optional): The genetic code of the automaton.
            name (optional): The name of the automaton. Arbitrary string identifier.
        """
        if "environment" not in kwargs:
            raise ValueError("AutomatonBase requires 'environment' in kwargs.")
        self.environment: Environment = kwargs["environment"]
        self.name = kwargs.get("name", "UnnamedAutomaton")
        self.genetic_code = kwargs.get("genetic_code", GeneticCodeDict({}))
        self.coords: list[int] = []  # n-dimensional coordinates.
        self.fitness: float = 0.0
        self.last_action: int = -1  # Last action taken.

    def attempt_action(self, action: int) -> ActionStatus:
        """Given an action integer, attempt to perform the corresponding action.

        The action is an attempt as the environment may not allow it (e.g. moving
        into a wall). The automaton can use the result of the action to update its
        internal state or make decisions in subsequent ticks.

        This method is intended to be overridden by subclasses to define how the
        automaton interacts with its environment based on the response generated
        by the tick method.

        Args:
            action: An integer representing the action to be taken.

        Returns:
            An ActionStatus enum value representing the result of the action.
        """
        self.last_action = action
        return ActionStatus.SUCCEEDED

    def reset(self) -> None:
        """Resets the automaton's state and fitness."""
        self.fitness = 0.0
        self.coords = []
        self.last_action = -1

    def tick(self) -> int:
        """Given the current environment stimulus, compute the response.

        Returns:
            An integer representing the response of the automaton.
        """
        raise NotImplementedError("Automaton.tick() must be implemented by subclasses.")


class AutomatonISBase(AutomatonBase):
    """Base class for an automaton that interacts with an environment and maintains
    internal state."""

    def __init__(self, **kwargs) -> None:
        """Initializes the automaton with a name and genetic code.

        The genetic code input is required to be:
            [0:state_len] → internal state
            [state_len:] → environment stimulus
        and the response:
            [:resp_len] → response to environment
            [resp_len:] → new internal state
        Args:
            name: The name of the automaton. Arbitrary string identifier.
            genetic_code: The genetic code of the automaton.
            env_bits: (int) The length of the environment byte string in bits.
            state_bits: (int) The length of the internal state byte string in bits.
            resp_bits: (int) The length of the response byte string in bits.
        """
        super().__init__(**kwargs)

        if "env_bits" not in kwargs:
            raise ValueError("AutomatonISBase requires 'env_bits' in kwargs.")
        if "state_bits" not in kwargs:
            raise ValueError("AutomatonISBase requires 'state_bits' in kwargs.")
        if "resp_bits" not in kwargs:
            raise ValueError("AutomatonISBase requires 'resp_bits' in kwargs.")
        self.env_bits = kwargs["env_bits"]
        self.env_bytes = (self.env_bits + 7) >> 3
        self.state_bits = kwargs["state_bits"]
        self.state_bytes = (self.state_bits + 7) >> 3
        self.resp_bits = kwargs["resp_bits"]
        self.resp_bytes = (self.resp_bits + 7) >> 3
        self.env_mask = (1 << self.env_bits) - 1
        self.state_mask = (1 << self.state_bits) - 1
        self.resp_mask = (1 << self.resp_bits) - 1
        self.internal_state: int = 0

    def reset(self) -> None:
        """Resets the automaton's internal state and fitness."""
        super().reset()
        self.internal_state = 0

    def tick(self) -> int:
        """Given the current environment stimulus, compute the response and update internal state.

        Returns:
            An integer representing the response of the automaton. The low
            ``state_bits`` of the genetic code output become the new internal
            state; the remaining high bits are returned as the response.
        """
        input_code = (self.internal_state << self.env_bits) | (
            self.environment.get_local(self.coords) & self.env_mask
        )
        output_code = self.genetic_code[input_code]
        self.internal_state = output_code & self.state_mask
        response = output_code >> self.state_bits
        return response
