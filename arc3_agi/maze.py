"""Simple maze environment for testing Automatons."""

from __future__ import annotations

from random import getrandbits, randrange
from typing import Optional, Tuple

from numpy import argwhere, int32, ones, uint8, uint16, zeros
from numpy.random import default_rng
from numpy.typing import NDArray
from terry2 import (
    INT32_ZERO,
    UINT16_ZERO,
    Automaton,
    Environment2DGrid,
    GeneticCode2DGrid,
)


def generate_maze(
    side_length_bits: int, seed: Optional[int] = None
) -> tuple[NDArray[uint8], NDArray[uint8]]:
    """Randomly generate a perfect maze on a square grid.

    Parameters
    ----------
    side_length_bits:
        Grid side length is ``2 ** side_length_bits``.  Must be >= 4 (i.e.
        the minimum grid side is 16).
    seed:
        Optional integer seed for reproducible results.

    Returns
    -------
    wall : NDArray[uint8], shape (side, side)
        1 = wall, 0 = free space.  Outer border is always 1.
    goal : NDArray[uint8], shape (side, side)
        Exactly one cell is 1 (the goal); all others are 0.

    Algorithm
    ---------
    Iterative DFS (recursive-backtracker) on an odd-indexed cell lattice.
    Every free cell belongs to a single connected region (perfect maze).
    Diagonal wall junctions are preserved — movement is strictly 4-directional.
    """
    if side_length_bits < 4:
        raise ValueError(
            f"side_length_bits must be >= 4 (minimum grid side 16), got {side_length_bits}"
        )

    rng = default_rng(seed)
    side = 2**side_length_bits

    # Start with all walls; free cells will be carved in.
    wall = ones((side, side), dtype=uint8)

    # Cell coordinates live on odd indices: 1, 3, 5, ..., side-2.
    # Number of cells per axis.
    n_cells = side // 2 - 1  # e.g. side=16 → 7 cells per axis

    # Visited flag indexed by (cell_row, cell_col) where cell coords are 0-based.
    visited = zeros((n_cells, n_cells), dtype=bool)

    def cell_to_grid(cr: int, cc: int) -> Tuple[int, int]:
        """Map cell index (0-based) to grid index."""
        return 2 * cr + 1, 2 * cc + 1

    # --- Cardinal neighbour directions (dr, dc) in cell-index space ---
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Choose a random starting cell.
    start_cr = int(rng.integers(0, n_cells))
    start_cc = int(rng.integers(0, n_cells))

    gr, gc = cell_to_grid(start_cr, start_cc)
    wall[gr, gc] = 0
    visited[start_cr, start_cc] = True

    # Iterative DFS stack holds cell indices.
    stack = [(start_cr, start_cc)]

    while stack:
        cr, cc = stack[-1]

        # Collect unvisited neighbours.
        neighbours = []
        for dr, dc in DIRS:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < n_cells and 0 <= nc < n_cells and not visited[nr, nc]:
                neighbours.append((nr, nc, dr, dc))

        if neighbours:
            # Pick one at random.
            idx = int(rng.integers(0, len(neighbours)))
            nr, nc, dr, dc = neighbours[idx]

            # Carve the passage: free the neighbour cell and the wall between.
            ngr, ngc = cell_to_grid(nr, nc)
            wall[ngr, ngc] = 0
            # The intermediate wall sits one step (in grid space) between the two cells.
            wall[gr + dr, gc + dc] = 0

            visited[nr, nc] = True
            stack.append((nr, nc))
            gr, gc = ngr, ngc
        else:
            stack.pop()
            if stack:
                cr, cc = stack[-1]
                gr, gc = cell_to_grid(cr, cc)

    # --- Place goal at a random free cell ---
    free_cells = argwhere(wall == 0)
    goal = zeros((side, side), dtype=uint8)
    chosen = free_cells[int(rng.integers(0, len(free_cells)))]
    goal[chosen[0], chosen[1]] = 1

    return wall, goal


class Maze(Environment2DGrid):
    """A simple maze environment for testing Automatons."""

    def __init__(self, side_length_bits: int = 6, seed: Optional[int] = None) -> None:
        super().__init__(side_length_bits)
        self.add_layer(self.LKEYS.WALL, mutable=False)
        self.add_layer(self.LKEYS.GOAL, mutable=False)

        wall, goal = generate_maze(side_length_bits, seed=seed)
        self.ilayers[self.LKEYS.WALL][:] = wall
        self.ilayers[self.LKEYS.GOAL][:] = goal

        self.free_cells = argwhere(wall == 0)


class MazeAutomaton(Automaton, radius=1, environment=Maze()):
    """Automaton that can navigate a maze environment."""

    def __init__(
        self,
        genetic_code: GeneticCode2DGrid,
        state: uint16,
        x: int32,
        y: int32,
        orientation: Automaton.Orientation,
    ) -> None:
        super().__init__(genetic_code, state, x, y, orientation)
        self.ticks_survived = 0
        self.bumps_into_wall = 0
        self.reached_goal = False
        self.fitness = 0.0

    def tick(self) -> None:
        """Perform one tick of the automaton's behavior."""
        super().tick()  # Get action from genetic code and update position/orientation.
        self.ticks_survived += 1
        if self.goal_layer[self.y, self.x] == 1:
            self.reached_goal = True
            # Simple fitness function: reward reaching the goal, and also surviving longer.
            self.fitness = 1000.0 / self.ticks_survived

    def move_forward(self) -> None:
        """Move forward in the current orientation if possible."""
        old_x = self.x
        old_y = self.y
        super().move_forward()
        # Check for wall collision; if collided, revert to old position.
        if self.environment.ilayers[self.environment.LKEYS.WALL][self.y, self.x] == 1:
            self.x = old_x
            self.y = old_y
            self.bumps_into_wall += 1


class Population:
    """Represents a population of automata for evolutionary processes."""

    def __init__(self, size: int) -> None:
        self.automata = [
            MazeAutomaton(
                genetic_code=GeneticCode2DGrid(),
                state=UINT16_ZERO,
                x=INT32_ZERO,
                y=INT32_ZERO,
                orientation=Automaton.Orientation(randrange(4)),
            )
            for _ in range(size)
        ]
        # All automata share the same maze environment.
        self.maze = self.automata[0].environment

    def tick(self) -> None:
        """Perform a tick for all automata in the population."""
        for automaton in self.automata:
            automaton.tick()

    def evolve(self) -> None:
        """Evolve the population based on some fitness function."""
        self.automata.sort(key=lambda a: a.fitness, reverse=True)
        # For simplicity, we can just keep the top 50% of the population and
        # replace the rest with offspring of the top performers.
        survivors = self.automata[: len(self.automata) // 2]
        offspring = []
        for i in range(len(self.automata) // 2):
            parent1 = survivors[randrange(len(survivors))]
            assert isinstance(parent1.genetic_code, GeneticCode2DGrid)
            parent2 = survivors[randrange(len(survivors))]
            assert isinstance(parent2.genetic_code, GeneticCode2DGrid)
            child_genetic_code = parent1.genetic_code.crossover(parent2.genetic_code)
            child = MazeAutomaton(
                genetic_code=child_genetic_code,
                state=uint16(getrandbits(16)),
                x=INT32_ZERO,
                y=INT32_ZERO,
                orientation=Automaton.Orientation(randrange(4)),
            )
            offspring.append(child)
        self.automata[len(self.automata) // 2 :] = offspring


if __name__ == "__main__":
    population = Population(size=100)
    rng = default_rng(42)
    maze = population.maze
    assert isinstance(maze, Maze)
    for i in population.automata:
        i.x, i.y = maze.free_cells[int(rng.integers(0, len(maze.free_cells)))]
    population.tick()
    population.evolve()
