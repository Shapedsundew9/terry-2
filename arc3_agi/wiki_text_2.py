from __future__ import annotations

from functools import lru_cache
from signal import SIGINT, signal
from typing import TYPE_CHECKING, Any, Protocol, cast

from arc3_agi.automaton import AutomatonISBase
from arc3_agi.checkpoint import CheckpointConfig
from arc3_agi.environment import ByteEnv
from arc3_agi.fingerprint import FingerprintConfig
from arc3_agi.genetic_code import GeneticCodeTsetlin
from arc3_agi.population import Population

if TYPE_CHECKING:
    from arc3_agi.population_rendering import PopulationGenerationSnapshot


TICKS_PER_GENERATION = 1000
GENERATIONS_PER_CHART_UPDATE = 10
WIKITEXT_DATASET_NAME = "Salesforce/wikitext"
WIKITEXT_DATASET_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_SPLIT = "train"


class PopulationCharts(Protocol):
    def update(
        self,
        snapshot: PopulationGenerationSnapshot,
        duration_s: float | None,
    ) -> None: ...


@lru_cache(maxsize=1)
def load_wikitext_environment(
    dataset_name: str = WIKITEXT_DATASET_NAME,
    dataset_config: str = WIKITEXT_DATASET_CONFIG,
    split: str = WIKITEXT_SPLIT,
) -> ByteEnv:
    """Load WikiText 2 on first use and return it as a byte environment."""
    from datasets import load_dataset

    wikitext = load_dataset(dataset_name, dataset_config)
    dataset_split = cast(Any, wikitext)[split]
    texts = [text for text in dataset_split["text"] if text]
    return ByteEnv(name="WikiEnv", array=texts, encoding="utf-8")


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
                num_clauses=kwargs.get("num_clauses", 16),
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
        self.coords = [self.rng.randrange(len(self.environment.get())), 0]
        self.remaining_bytes = 0  # Reset remaining bytes to read.
        self.right = 0
        self.total = 0
        self.fitness = 0.0

    def tick(self, **kwargs) -> int:
        """Perform a single tick of the automaton."""
        texts = self.environment.get()
        if not self.remaining_bytes:
            if self.coords[0] < len(texts) - 1:
                self.coords[0] += 1  # Move to the next bytes object in the environment.
            else:
                self.coords[0] = 0
            self.coords[1] = 0  # Reset the byte index to 0 for the new bytes object.
            self.remaining_bytes = len(texts[self.coords[0]])
        prediction = super().tick(**kwargs)
        self.total += 1  # Increment the total number of predictions.
        self.remaining_bytes -= 1  # Decrement the remaining bytes to read.
        self.coords[1] += 1  # Move to the next byte in the current bytes object.
        actual = (
            0 if not self.remaining_bytes else texts[self.coords[0]][self.coords[1]]
        )
        if prediction == actual:
            self.right += 1  # Increment the count of correct predictions.
        self.fitness = self.right / self.total
        return prediction


class WikiPopulation(Population):
    def __init__(self, **kwargs) -> None:
        environment = kwargs.get("environment")
        if environment is None:
            environment = load_wikitext_environment()
        super().__init__(
            AutomatonClass=kwargs.get("automaton_class", WikiAutomaton),
            environment=environment,
            size=kwargs.get("size", 100),
            seed=kwargs.get("seed", None),
            automaton_params=kwargs.get("automaton_params", None),
            checkpoint_config=kwargs.get("checkpoint_config", None),
            fingerprint_config=kwargs.get("fingerprint_config", None),
        )
        assert isinstance(
            self.environment, ByteEnv
        ), "WikiPopulation requires a ByteEnv environment."


def run_charted_generation(
    population: Population,
    charts: PopulationCharts,
    ticks_per_generation: int = TICKS_PER_GENERATION,
    update_chart: bool = True,
) -> PopulationGenerationSnapshot:
    """Evaluate, evolve, and render one population generation."""
    from arc3_agi.population_rendering import PopulationGenerationSnapshot

    population.run_generation(ticks_per_generation)
    snapshot = PopulationGenerationSnapshot.capture(population.automata)
    population.evolve()
    if update_chart:
        charts.update(snapshot, population.fitness_history[-1].get("duration_s"))
    return snapshot


def run_live() -> None:
    """Evolve WikiText predictors and display live population charts."""
    import traceback

    import matplotlib

    matplotlib.use("webagg")
    import matplotlib.pyplot as plt

    from arc3_agi.population_rendering import PopulationChartSuite

    environment = load_wikitext_environment()
    charts = PopulationChartSuite()
    population = WikiPopulation(
        environment=environment,
        checkpoint_config=CheckpointConfig(enabled=False),
        fingerprint_config=FingerprintConfig(bits=4, tournament_k=4),
    )
    generations_per_chart_update = max(1, GENERATIONS_PER_CHART_UPDATE)
    stopped = False

    def stop() -> None:
        nonlocal stopped
        if stopped:
            return
        stopped = True
        timer.stop()
        charts.close()

    def simulation_step() -> None:
        if stopped:
            return
        try:
            # Collect each generation so chart history stays full-fidelity,
            # then render once per batch to reduce UI overhead.
            pending_snapshots = []
            pending_durations = []
            for _ in range(generations_per_chart_update):
                snapshot = run_charted_generation(
                    population,
                    charts,
                    update_chart=False,
                )
                pending_snapshots.append(snapshot)
                pending_durations.append(
                    population.fitness_history[-1].get("duration_s")
                )
            charts.update_batch(pending_snapshots, pending_durations)
        except Exception:
            traceback.print_exc()
            stop()

    timer = charts.canvas.new_timer(interval=1)
    timer.add_callback(simulation_step)
    timer.start()

    def handle_close(event) -> None:
        stop()

    def handle_sigint(sig, frame) -> None:
        stop()

    for figure in charts.figures:
        figure.canvas.mpl_connect("close_event", handle_close)
    signal(SIGINT, handle_sigint)

    try:
        plt.show()
    finally:
        stop()


if __name__ == "__main__":
    run_live()
