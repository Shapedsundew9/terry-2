from __future__ import annotations

from enum import IntEnum

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
            name: The name of the automaton. Arbitrary string identifier.
            genetic_code: The genetic code of the automaton.
        """
        self.name = kwargs.get("name", "UnnamedAutomaton")
        self.genetic_code = kwargs.get("genetic_code", GeneticCodeDict({}))
        self.coords: list[int] = []
        self.fitness: float = 0.0

    def tick(self, environment: bytes) -> bytes:
        """Given the current environment stimulus, compute the response.

        Args:
            environment: A byte string representing the current stimulus from the environment.

        Returns:
            A byte string representing the response of the automaton.
        """
        raise NotImplementedError("Automaton.tick() must be implemented by subclasses.")


class AutomatonISBase(AutomatonBase):
    """Base class for an automaton that interacts with an environment and maintains
    internal state."""

    def __init__(self, **kwargs) -> None:
        """Initializes the automaton with a name and genetic code.

        The genetic code input is required to be:
            [0:state_len] → internal state
            [state_len:state_len+env_len] → environment stimulus
        and the respnse:
            [0:state_len] → new internal state
            [state_len:state_len+resp_len] → response to environment
        Args:
            name: The name of the automaton. Arbitrary string identifier.
            genetic_code: The genetic code of the automaton.
            env_len: (int) The length of the environment byte string.
            state_len: (int) The length of the internal state byte string.
            resp_len: (int) The length of the response byte string.
        """
        super().__init__(**kwargs)

        if "env_len" not in kwargs:
            raise ValueError("AutomatonISBase requires 'env_len' in kwargs.")
        if "state_len" not in kwargs:
            raise ValueError("AutomatonISBase requires 'state_len' in kwargs.")
        if "resp_len" not in kwargs:
            raise ValueError("AutomatonISBase requires 'resp_len' in kwargs.")
        self.env_len = kwargs["env_len"]
        self.state_len = kwargs["state_len"]
        self.resp_len = kwargs["resp_len"]
        self.internal_state: bytes = bytes(
            self.state_len
        )  # initialize internal state to all zeros

    def tick(self, environment: bytes) -> bytes:
        """Given the current environment stimulus, compute the response and update internal state.

        Args:
            environment: A byte string representing the current stimulus from the environment.

        Returns:
            A byte string representing the response of the automaton.
        """
        input_code = self.internal_state + environment
        output_code = self.genetic_code[input_code]
        self.internal_state = output_code[: self.state_len]
        response = output_code[self.state_len : self.state_len + self.resp_len]
        return response

    def attempt_action(self, action: bytes) -> ActionStatus:
        """Given an action byte string, attempt to perform the corresponding action.

        The action is an attempt as the environment may not allow it (e.g. moving
        into a wall). The automaton can use the result of the action to update its
        internal state or make decisions in subsequent ticks.

        This method is intended to be overridden by subclasses to define how the
        automaton interacts with its environment based on the response generated
        by the tick method.

        Args:
            action: A byte string representing the action to be taken.

        Returns:
            An ActionStatus enum value representing the result of the action.
        """
        raise NotImplementedError(
            "AutomatonISBase.attempt_action() must be implemented by subclasses."
        )
