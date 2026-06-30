from random import randrange

from arc3_agi.automaton import AutomatonBase
from arc3_agi.environment import Environment
from arc3_agi.genetic_code import GeneticCode


class Population:
    """Represents a population of automata for evolutionary processes."""

    def __init__(
        self, size: int, AutomatonClass: type[AutomatonBase], environment: Environment
    ) -> None:
        self._automata_class = AutomatonClass
        self.automata = [AutomatonClass(environment=environment) for _ in range(size)]
        self.environment = environment
        self.tick_count = 0

    def tick(self) -> None:
        """Perform a tick for all automata using batched environment observation."""
        for automaton in self.automata:
            automaton.tick()
        self.tick_count += 1

    def evolve(self) -> list[float]:
        """Evolve the population based on some fitness function."""
        self.automata.sort(key=lambda a: a.fitness, reverse=True)
        # For simplicity, we can just keep the top 50% of the population and
        # replace the rest with offspring of the top performers.
        survivors = self.automata[: len(self.automata) // 2]
        offspring = []
        for i in range(len(self.automata) // 2):
            parent1 = survivors[randrange(len(survivors))]
            assert isinstance(parent1.genetic_code, GeneticCode)
            parent2 = survivors[randrange(len(survivors))]
            assert isinstance(parent2.genetic_code, GeneticCode)
            child_genetic_code = parent1.genetic_code.crossover(parent2.genetic_code)
            child = self._automata_class(
                genetic_code=child_genetic_code, environment=self.environment
            )
            offspring.append(child)
        fitnesses = [a.fitness for a in self.automata]
        self.automata[len(self.automata) // 2 :] = offspring
        for a in self.automata:
            a.reset()
        return fitnesses
