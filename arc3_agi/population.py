from arc3_agi.automaton import AutomatonBase, AutomatonISBase
from arc3_agi.environment import Environment
from arc3_agi.genetic_code import GeneticCode, GeneticCodeDict
from arc3_agi.maze import Maze, MazeAutomaton


class Population:
    """Represents a population of automata for evolutionary processes."""

    def __init__(
        self, size: int, AutomatonClass: type[AutomatonBase], environment: Environment
    ) -> None:
        self.automata = [AutomatonClass() for _ in range(size)]
        self.environment = environment

    def tick(self) -> None:
        """Perform a tick for all automata using batched environment observation."""
        for automaton in self.automata:
            automaton.tick(self.environment.get_local(automaton.coords))

    def evolve(self) -> list[float]:
        """Evolve the population based on some fitness function."""
        for a in self.automata:
            # Simple fitness function: prioritize reaching the goal, then surviving longer,
            # then fewer wall bumps, then more moves.
            a.fitness = a.fitness + (
                -1.0 * a.bumps_into_wall + 1.0 * a.num_moves - 1.0 * a.backtracks
            )
        self.automata.sort(key=lambda a: a.fitness, reverse=True)
        # For simplicity, we can just keep the top 50% of the population and
        # replace the rest with offspring of the top performers.
        survivors = self.automata[: len(self.automata) // 2]
        offspring = []
        for i in range(len(self.automata) // 2):
            parent1 = survivors[randrange(len(survivors))]
            # parent1.start_position()  # Teleport parent to a new random free cell for the next run.
            assert isinstance(parent1.genetic_code, GeneticCode2DGrid)
            parent2 = survivors[randrange(len(survivors))]
            # parent2.start_position()  # Teleport parent to a new random free cell for the next run.
            assert isinstance(parent2.genetic_code, GeneticCode2DGrid)
            child_genetic_code = parent1.genetic_code.crossover(parent2.genetic_code)
            child = MazeAutomaton(
                genetic_code=child_genetic_code,
                state=0,
                x=0,
                y=0,
                orientation=Automaton.Orientation(randrange(4)),
            )
            offspring.append(child)
        fitnesses = [a.fitness for a in self.automata]
        self.automata[len(self.automata) // 2 :] = offspring
        for a in self.automata:
            a.reset_stats()
            a.fitness = 0.0
        return fitnesses
