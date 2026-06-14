"""Simple maze environment for testing Automatons."""

from __future__ import annotations

from random import getrandbits, randrange
from typing import Optional, Tuple

import pygame
from numpy import argwhere, int32, ones, uint8, uint16, zeros
from numpy.random import default_rng
from numpy.typing import NDArray

from arc3_agi.terry2 import (
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


class MazeRenderer:
    """Simple pygame renderer for the Maze environment."""

    _COLOR_WALL = (255, 255, 255)
    _COLOR_FREE = (0, 0, 0)
    _COLOR_GOAL = (255, 215, 0)
    _COLOR_AUTOMATON = (0, 255, 255)
    _COLOR_TRIANGLE = (0, 80, 180)

    def __init__(self, maze: Maze, cell_size: int = 8) -> None:
        self.maze = maze
        self.cell_size = cell_size
        self.width = maze.width * cell_size
        self.height = maze.height * cell_size
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Maze Viewer")
        self._bg = self._build_background()

    def _build_background(self) -> pygame.Surface:
        surface = pygame.Surface((self.width, self.height))
        wall_layer = self.maze.ilayers[self.maze.LKEYS.WALL]
        goal_layer = self.maze.ilayers[self.maze.LKEYS.GOAL]
        cs = self.cell_size
        for row in range(self.maze.height):
            for col in range(self.maze.width):
                px, py = col * cs, row * cs
                if goal_layer[row, col] == 1:
                    color = self._COLOR_GOAL
                elif wall_layer[row, col] == 1:
                    color = self._COLOR_WALL
                else:
                    color = self._COLOR_FREE
                surface.fill(color, (px, py, cs, cs))
        return surface

    def _draw_automaton(
        self, surface: pygame.Surface, automaton: "MazeAutomaton"
    ) -> None:
        cs = self.cell_size
        col, row = int(automaton.x), int(automaton.y)
        px, py = col * cs, row * cs
        # Fill cell with automaton color
        surface.fill(self._COLOR_AUTOMATON, (px, py, cs, cs))
        # Draw direction triangle
        half = cs // 2
        margin = max(1, cs // 6)
        orientation = automaton.orientation
        if orientation == automaton.Orientation.UP:
            pts = [
                (px + half, py + margin),
                (px + margin, py + cs - margin),
                (px + cs - margin, py + cs - margin),
            ]
        elif orientation == automaton.Orientation.DOWN:
            pts = [
                (px + half, py + cs - margin),
                (px + margin, py + margin),
                (px + cs - margin, py + margin),
            ]
        elif orientation == automaton.Orientation.LEFT:
            pts = [
                (px + margin, py + half),
                (px + cs - margin, py + margin),
                (px + cs - margin, py + cs - margin),
            ]
        else:  # RIGHT
            pts = [
                (px + cs - margin, py + half),
                (px + margin, py + margin),
                (px + margin, py + cs - margin),
            ]
        pygame.draw.polygon(surface, self._COLOR_TRIANGLE, pts)

    def render(self, automata: list) -> None:
        surface = self._bg.copy()
        for automaton in automata:
            self._draw_automaton(surface, automaton)
        self.screen.blit(surface, (0, 0))
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()


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
    FPS = 10
    TICKS_PER_EVOLVE = 100

    population = Population(size=100)
    rng = default_rng(42)
    maze = population.maze
    assert isinstance(maze, Maze)
    for automaton in population.automata:
        automaton.x, automaton.y = maze.free_cells[
            int(rng.integers(0, len(maze.free_cells)))
        ]

    renderer = MazeRenderer(maze)
    clock = pygame.time.Clock()
    tick_count = 0
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            if not running:
                break

            population.tick()
            tick_count += 1

            if tick_count % TICKS_PER_EVOLVE == 0:
                population.evolve()
                # Reposition the newly-created offspring (bottom half) to random free cells
                half = len(population.automata) // 2
                for automaton in population.automata[half:]:
                    automaton.x, automaton.y = maze.free_cells[
                        int(rng.integers(0, len(maze.free_cells)))
                    ]

            renderer.render(population.automata)
            clock.tick(FPS)
    finally:
        renderer.close()
