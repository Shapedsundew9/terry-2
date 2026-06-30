"""This module contains the Parethesis class, which is an environment
where each element is a parenthesis and the state of the environment
is represented by a sequence of parentheses that is correctly closed."""

from enum import IntEnum
from typing import Optional

from arc3_agi.automaton import ActionStatus, AutomatonISBase
from arc3_agi.environment import Int1DArray
from arc3_agi.genetic_code import GeneticCodeDict
from arc3_agi.population import Population


class ParenthesisToken(IntEnum):
    # BOS = 0
    OPEN = 1  # Opening parenthesis '('
    CLOSE = 2  # Closing parenthesis ')'


class ParenthesisAction(IntEnum):
    CONSISTENT = 0  # Sequence is consistent at this point (can still be valid)
    VALID = 1  # Sequence is valid at this point (correctly closed so far)
    INVALID = 2  # Sequence is invalid at this point (incorrectly closed)
    DONE = 3  # Sequence is done (end of sequence reached)


class Parenthesis(Int1DArray):
    """The Parenthesis class is an environment where each element is a parenthesis
    and the state of the environment is represented by a sequence of parentheses
    that is correctly closed.

    The environment is represented as a 1D array of integers, where each integer
    represents a parenthesis. The value 1 represents an opening parenthesis '(',
    and the value 2 represents a closing parenthesis ')'.

    NOTE: The value 0 is
    reserved for the beginning of the sequence (BOS) and is not used
    in the sequence itself as the Automaton has a starting state and there is no
    need to look backwards."""

    def __init__(self, **kwargs) -> None:
        """Initializes the Parenthesis environment.

        Args:
            length (int): The length of the sequence of parentheses.
                Default is 10.
            valid: (bool): If True, generates a valid sequence of parentheses.
                If False, generates an invalid sequence of parentheses.
                Default is True.

        """
        super().__init__(
            name="Parenthesis",
            description="An environment where each element is a parenthesis and"
            " the state of the environment is represented by a sequence "
            "of parentheses that is correctly closed.",
        )
        length = kwargs.get("length", 10)
        valid = kwargs.get("valid", True)
        if valid:
            self._array = self.generate_valid_sequence(length)
        else:
            self._array = self.generate_invalid_sequence(length)
        self._validation_state = self.validation_state(self._array)

    def is_end(self, x: int) -> bool:
        """Checks if the given index is at the end of the sequence.

        Args:
            x (int): The index to check.

        Returns:
            bool: True if the index is at the end of the sequence, False otherwise.
        """
        return x >= len(self._array)

    def is_valid(self, sequence: Optional[list[int]] = None) -> bool:
        """Checks if the given sequence of parentheses is valid (correctly closed).

        Args:
            sequence (Optional[list[int]]): A list of integers representing a sequence of parentheses.
                If None, the current environment's array will be used.

        Returns:
            bool: True if the sequence is valid (correctly closed), False otherwise.
        """
        if sequence is None:
            sequence = self._array
        balance = 0
        for token in sequence:
            if token == ParenthesisToken.OPEN:
                balance += 1
            elif token == ParenthesisToken.CLOSE:
                balance -= 1
            # If at any point the balance is negative, the sequence is invalid
            if balance < 0:
                return False
        # The sequence is valid if the balance is zero at the end
        return balance == 0

    def generate_valid_sequence(self, length: int) -> list[int]:
        """Generates a random sequence of parentheses of the given length.

        Args:
            length (int): The length of the sequence to generate.

        Returns:
            list[int]: A list of integers representing a random sequence of parentheses.
        """
        import random

        sequence: list[int] = []
        balance = 0
        for _ in range(length):
            if balance == 0:
                # Must add an opening parenthesis
                sequence.append(ParenthesisToken.OPEN.value)
                balance += 1
            else:
                # Randomly choose to add an opening or closing parenthesis
                if random.choice([True, False]):
                    sequence.append(ParenthesisToken.OPEN.value)
                    balance += 1
                else:
                    sequence.append(ParenthesisToken.CLOSE.value)
                    balance -= 1
        # If there are unmatched opening parentheses, close them
        while balance > 0:
            sequence.append(ParenthesisToken.CLOSE.value)
            balance -= 1
        return sequence

    def generate_invalid_sequence(self, length: int) -> list[int]:
        """Generates a random invalid sequence of parentheses of the given length.

        Args:
            length (int): The length of the sequence to generate.

        Returns:
            list[int]: A list of integers representing a random invalid sequence of parentheses.
        """
        import random

        sequence: list[int] = []
        balance = 0
        for _ in range(length):
            # Randomly choose to add an opening or closing parenthesis
            if random.choice([True, False]):
                sequence.append(ParenthesisToken.OPEN.value)
                balance += 1
            else:
                sequence.append(ParenthesisToken.CLOSE.value)
                balance -= 1
        # If the random sequence turns out to be valid, we can make it
        # invalid by inverting the last parenthesis
        if balance == 0:
            if sequence[-1] == ParenthesisToken.OPEN.value:
                sequence[-1] = ParenthesisToken.CLOSE.value
            else:
                sequence[-1] = ParenthesisToken.OPEN.value
        return sequence

    def validation_state(
        self, sequence: Optional[list[int]] = None
    ) -> list[ParenthesisAction]:
        """Returns a list of ParenthesisAction indicating the validity of the sequence at each point.

        Args:
            sequence (Optional[list[int]]): A list of integers representing a sequence of parentheses.
                If None, the current environment's array will be used.
        Returns:
            list[ParenthesisAction]: A list of ParenthesisAction indicating the validity of the
                sequence at each point.
        """
        if sequence is None:
            sequence = self._array
        actions: list[ParenthesisAction] = []
        balance = 0
        for token in sequence:
            if token == ParenthesisToken.OPEN:
                balance += 1
            elif token == ParenthesisToken.CLOSE:
                balance -= 1
            # Determine the action based on the current balance
            if balance < 0:
                actions.append(ParenthesisAction.INVALID)
            elif balance == 0:
                actions.append(ParenthesisAction.VALID)
            else:
                actions.append(ParenthesisAction.CONSISTENT)
        return actions


class ParenthesisAutomaton(AutomatonISBase):
    """An automaton that interacts with the Parethesis environment."""

    def __init__(self, **kwargs) -> None:
        """Initializes the ParethesisAutomaton.

        Args:
            name (str): The name of the automaton.
            genetic_code (bytes): The genetic code of the automaton.
            env_bits (int): The length of the environment byte string in bits.
            state_bits (int): The length of the internal state byte string in bits.
            resp_bits (int): The length of the response byte string in bits.
        """
        state_bits = 8
        resp_bits = 2
        if "genetic_code" not in kwargs:
            genetic_code = GeneticCodeDict({}, resp_bits=resp_bits + state_bits)
            kwargs["genetic_code"] = genetic_code
        super().__init__(env_bits=8, state_bits=8, resp_bits=2, **kwargs)
        self.coords = [kwargs.get("x", 0)]
        assert isinstance(
            self.environment, Parenthesis
        ), "ParenthesisAutomaton requires a Parenthesis environment."

    def attempt_action(self, action: int) -> ActionStatus:
        """Attempts to perform the action specified by the automaton's response.

        Args:
            action (int): An integer representing the action to be performed.
        """
        assert isinstance(
            self.environment, Parenthesis
        ), "ParenthesisAutomaton requires a Parenthesis environment."
        action &= self.resp_mask
        if action == self.environment._validation_state[self.coords[0]].value:
            self.fitness += 1.0  # Reward for correct action
        else:
            self.fitness -= 1.0  # Penalty for incorrect action
        if action == ParenthesisAction.INVALID.value:
            return ActionStatus.SUCCEEDED
        # In the case the sequence is valid or consistent, we can move to the next
        # token in the environment. For now, we assume all actions are successful.
        self.coords[0] += 1
        return (
            ActionStatus.SUCCEEDED
            if not self.environment.is_end(self.coords[0])
            else ActionStatus.FAILED
        )

    def reset(self) -> None:
        """Resets the automaton's state and position in the sequence."""
        super().reset()
        self.coords = [0]

    def tick(self) -> int:
        """Perform a tick of the automaton."""
        # super().tick() already updates internal_state and returns only the action bytes.
        action = super().tick()
        action_status = self.attempt_action(action)
        if action_status == ActionStatus.FAILED:
            self.last_action = ParenthesisAction.DONE.value
            return ParenthesisAction.DONE.value
        return action


if __name__ == "__main__":
    # Example usage

    length = 10
    for _ in range(5):
        best_fitness = 0.0
        parenthesis_env = Parenthesis(length=length)
        population = Population(
            size=100, AutomatonClass=ParenthesisAutomaton, environment=parenthesis_env
        )
        while best_fitness < length:
            while not any(
                a.last_action == ParenthesisAction.DONE.value
                for a in population.automata
            ):
                population.tick()
            fitnesses = population.evolve()
            best_fitness = max(fitnesses)
            print(
                f"\rBest fitness in this generation: {best_fitness:<10}",
                end="",
                flush=True,
            )
        print("Found a solution!")
