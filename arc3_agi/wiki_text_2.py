from random import randrange

from datasets import load_dataset

from arc3_agi.automaton import AutomatonISBase
from arc3_agi.environment import ByteEnv
from arc3_agi.genetic_code import GeneticCodeTsetlin
from arc3_agi.population import Population

# Option 1: Smallest general English text (~4.5 MB)
wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
wikienv = ByteEnv(
    name="WikiEnv",
    array=[
        wikitext["train"][i]["text"]
        for i in range(len(wikitext["train"]))
        if wikitext["train"][i]["text"]
    ],  # Use only non-empty strings
    encoding="utf-8",
)


class WikiAutomaton(AutomatonISBase):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            name=kwargs.get("name", "Terry-2"),
            genetic_code=kwargs.get("genetic_code", None),
            env_bits=kwargs.get("env_bits", 16),
            state_bits=kwargs.get("state_bits", 8),
            resp_bits=kwargs.get("resp_bits", 8),
            environment=kwargs.get("environment"),
            fingerprint_config=kwargs.get("fingerprint_config", None),
            seed=kwargs.get("seed", None),
        )
        if self.genetic_code is None:
            self.genetic_code = GeneticCodeTsetlin(
                None,
                seed=self.rng.randint(0, 2**32 - 1),
                resp_bits=self.state_bits + self.resp_bits,
                input_bits=self.env_bits + self.state_bits,
            )
        assert isinstance(
            self.environment, ByteEnv
        ), "WikiAutomaton requires a ByteEnv environment."
        self.coords = [
            0,
            0,
        ]  # Initialize coordinates for the automaton in the environment.
        self.remaining_bytes = (
            0  # Initialize remaining bytes to read from the environment.
        )
        self.right = 0
        self.total = 0
        self.fitness = 0.0

    def reset(self) -> None:
        """Reset the automaton's state and coordinates."""
        super().reset()
        self.coords = [randrange(len(self.environment.get())), 0]
        self.remaining_bytes = 0  # Reset remaining bytes to read.
        self.right = 0
        self.total = 0
        self.fitness = 0.0

    def tick(self, **kwargs) -> int:
        """Perform a single tick of the automaton."""
        if not self.remaining_bytes:
            b = self.environment.get()
            if self.coords[0] < len(b) - 1:
                self.coords[0] += 1  # Move to the next bytes object in the environment.
            else:
                self.coords[0] = 0
            self.coords[1] = 0  # Reset the byte index to 0 for the new bytes object.
            self.remaining_bytes = len(b[self.coords[0]])
        prediction = super().tick(**kwargs)
        self.total += 1  # Increment the total number of predictions.
        self.remaining_bytes -= 1  # Decrement the remaining bytes to read.
        self.coords[1] += 1  # Move to the next byte in the current bytes object.
        actual = (
            0 if not self.remaining_bytes else self.environment.get_local(self.coords)
        )
        if prediction == actual:
            self.right += 1  # Increment the count of correct predictions.
        self.fitness = self.right / self.total
        return prediction


class WikiPopulation(Population):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            AutomatonClass=kwargs.get("automaton_class", WikiAutomaton),
            environment=kwargs.get("environment", wikienv),
            size=kwargs.get("size", 100),
            seed=kwargs.get("seed", None),
            automaton_params=kwargs.get("automaton_params", None),
            checkpoint_config=kwargs.get("checkpoint_config", None),
        )
        assert isinstance(
            self.environment, ByteEnv
        ), "WikiPopulation requires a ByteEnv environment."


if __name__ == "__main__":
    pop = WikiPopulation()
    for _ in range(10):
        for _ in range(1000):
            pop.tick()
        pop.evolve()
