from types import SimpleNamespace

import pytest

from arc3_agi.checkpoint import CheckpointConfig
from arc3_agi.environment import ByteEnv
from arc3_agi.fingerprint import FingerprintConfig
from arc3_agi.genetic_code import GeneticCodeDict
from arc3_agi.wiki_text_2 import (
    WikiAutomaton,
    WikiPopulation,
    load_wikitext_environment,
    run_charted_generation,
)


def test_injected_environment_does_not_load_dataset(monkeypatch) -> None:
    import arc3_agi.wiki_text_2 as wiki_text_2

    def fail_loader() -> ByteEnv:
        pytest.fail("default WikiText dataset should not be loaded")

    monkeypatch.setattr(wiki_text_2, "load_wikitext_environment", fail_loader)
    environment = ByteEnv(name="test", array=["abc"])

    population = WikiPopulation(
        environment=environment,
        size=2,
        checkpoint_config=CheckpointConfig(enabled=False),
    )

    assert population.environment is environment


def test_dataset_loader_filters_empty_text(monkeypatch) -> None:
    import datasets

    load_wikitext_environment.cache_clear()
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *args, **kwargs: {"train": {"text": ["", "abc", "def"]}},
    )

    environment = load_wikitext_environment()

    assert environment.get() == [b"abc", b"def"]
    load_wikitext_environment.cache_clear()


def test_dataset_loader_forwards_coordinates_and_caches(monkeypatch) -> None:
    import datasets

    calls: list[tuple[str, str]] = []

    def fake_load_dataset(name: str, config: str):
        calls.append((name, config))
        return {"validation": {"text": ["custom"]}}

    load_wikitext_environment.cache_clear()
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    first = load_wikitext_environment("example/wiki", "raw", "validation")
    second = load_wikitext_environment("example/wiki", "raw", "validation")

    assert first is second
    assert first.get() == [b"custom"]
    assert calls == [("example/wiki", "raw")]
    load_wikitext_environment.cache_clear()


def test_population_forwards_fingerprint_configuration() -> None:
    config = FingerprintConfig(bits=4, tournament_k=2)
    population = WikiPopulation(
        environment=ByteEnv(name="test", array=["abc"]),
        size=4,
        checkpoint_config=CheckpointConfig(enabled=False),
        fingerprint_config=config,
    )

    assert all(
        automaton.fingerprint is not None and automaton.fingerprint.bits == config.bits
        for automaton in population.automata
    )


def test_automaton_scores_next_raw_byte_and_end_sentinel() -> None:
    environment = ByteEnv(name="test", array=["abc"])
    inputs_to_predictions = {
        ord("a"): ord("b") << 8,
        (ord("a") << 8) | ord("b"): ord("c") << 8,
        (ord("b") << 8) | ord("c"): 0,
    }
    automaton = WikiAutomaton(
        environment=environment,
        genetic_code=GeneticCodeDict(inputs_to_predictions, resp_bits=16),
        seed=0,
    )

    assert [automaton.tick() for _ in range(3)] == [ord("b"), ord("c"), 0]
    assert automaton.right == 3
    assert automaton.total == 3
    assert automaton.fitness == 1.0


def test_run_charted_generation_snapshots_before_evolution() -> None:
    automaton = SimpleNamespace(fitness=0.0, fingerprint=None)

    class FakePopulation:
        def __init__(self) -> None:
            self.automata = [automaton]
            self.fitness_history: list[dict] = []
            self.ticks_per_generation = None

        def run_generation(self, ticks_per_generation: int) -> None:
            self.ticks_per_generation = ticks_per_generation
            automaton.fitness = 0.75

        def evolve(self) -> list[float]:
            self.fitness_history.append({"duration_s": 0.25})
            automaton.fitness = 0.0
            return [0.75]

    class FakeCharts:
        def __init__(self) -> None:
            self.snapshot = None
            self.duration_s = None

        def update(self, snapshot, duration_s) -> None:
            self.snapshot = snapshot
            self.duration_s = duration_s

    population = FakePopulation()
    charts = FakeCharts()

    snapshot = run_charted_generation(population, charts, ticks_per_generation=12)

    assert population.ticks_per_generation == 12
    assert snapshot.fitnesses == (0.75,)
    assert charts.snapshot is snapshot
    assert charts.duration_s == 0.25
