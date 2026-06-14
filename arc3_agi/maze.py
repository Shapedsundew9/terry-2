"""Simple maze environment for testing Automatons."""

from __future__ import annotations

from calendar import EPOCH
from random import getrandbits, randrange
from typing import Optional, Tuple

import matplotlib

matplotlib.use("webagg")
import matplotlib.pyplot as plt
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
        self.rng = default_rng(seed)


class MazeRenderer:
    """Simple matplotlib renderer for the Maze environment."""

    def __init__(self, maze: Maze, cell_size: int = 8) -> None:
        self.maze = maze
        self.cs = cell_size
        px_w = maze.width * cell_size
        px_h = maze.height * cell_size
        self.fig, self.ax = plt.subplots(figsize=(px_w / 100, px_h / 100), dpi=100)
        self.ax.set_axis_off()
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._bg = self._build_background()
        self._im = self.ax.imshow(self._bg, origin="upper", interpolation="nearest")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _build_background(self):
        cs = self.cs
        h, w = self.maze.height, self.maze.width
        rgb = zeros((h * cs, w * cs, 3), dtype=uint8)  # black = free
        wall_layer = self.maze.ilayers[self.maze.LKEYS.WALL]
        goal_layer = self.maze.ilayers[self.maze.LKEYS.GOAL]
        wall_up = wall_layer.repeat(cs, axis=0).repeat(cs, axis=1)
        goal_up = goal_layer.repeat(cs, axis=0).repeat(cs, axis=1)
        rgb[wall_up == 1] = [255, 255, 255]  # walls → white
        rgb[goal_up == 1] = [255, 215, 0]  # goal  → gold
        return rgb

    def _draw_triangle(self, frame, px0: int, py0: int, cs: int, orientation) -> None:
        half = cs // 2
        m = max(1, cs // 6)
        Ori = type(orientation)
        if orientation == Ori.UP:
            pts = [
                (px0 + half, py0 + m),
                (px0 + m, py0 + cs - m),
                (px0 + cs - m, py0 + cs - m),
            ]
        elif orientation == Ori.DOWN:
            pts = [
                (px0 + half, py0 + cs - m),
                (px0 + m, py0 + m),
                (px0 + cs - m, py0 + m),
            ]
        elif orientation == Ori.LEFT:
            pts = [
                (px0 + m, py0 + half),
                (px0 + cs - m, py0 + m),
                (px0 + cs - m, py0 + cs - m),
            ]
        else:  # RIGHT
            pts = [
                (px0 + cs - m, py0 + half),
                (px0 + m, py0 + m),
                (px0 + m, py0 + cs - m),
            ]
        (ax_, ay), (bx, by), (cx, cy) = pts
        x0, x1 = min(ax_, bx, cx), max(ax_, bx, cx)
        y0, y1 = min(ay, by, cy), max(ay, by, cy)
        denom = (by - cy) * (ax_ - cx) + (cx - bx) * (ay - cy)
        if denom == 0:
            return
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                u = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / denom
                v = ((cy - ay) * (x - cx) + (ax_ - cx) * (y - cy)) / denom
                if u >= 0 and v >= 0 and (1 - u - v) >= 0:
                    frame[y, x] = [0, 80, 180]

    def render(self, automata: list) -> None:
        frame = self._bg.copy()
        cs = self.cs
        for automaton in automata:
            row, col = int(automaton.y), int(automaton.x)
            py0, px0 = row * cs, col * cs
            frame[py0 : py0 + cs, px0 : px0 + cs] = [0, 255, 255]  # cyan
            self._draw_triangle(frame, px0, py0, cs, automaton.orientation)
        self._im.set_data(frame)
        self.fig.canvas.draw_idle()

    def is_open(self) -> bool:
        return plt.fignum_exists(self.fig.number)

    def close(self) -> None:
        plt.close(self.fig)


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
        self.bumps_into_wall = 0
        self.num_moves = 0
        self.fitness = 0.0
        self.start_position()

    def reset_stats(self) -> None:
        self.bumps_into_wall = 0
        self.num_moves = 0

    def start_position(self) -> None:
        assert isinstance(self.environment, Maze)
        pos = self.environment.free_cells[
            int(self.environment.rng.integers(0, len(self.environment.free_cells)))
        ]
        self.y, self.x = int32(pos[0]), int32(pos[1])

    def tick(self) -> None:
        """Perform one tick of the automaton's behavior."""
        super().tick()  # Get action from genetic code and update position/orientation.
        if self.goal_layer[self.y, self.x] == 1:
            self.reset_stats()  # Reset stats for the next run.
            self.fitness += 100.0  # Large reward for reaching the goal.
            self.start_position()  # Teleport to a new random free cell.

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
        else:
            self.num_moves += 1


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
        for a in self.automata:
            # Simple fitness function: prioritize reaching the goal, then surviving longer,
            # then fewer wall bumps, then more moves.
            a.fitness = a.fitness + (-1.0 * a.bumps_into_wall + 1.0 * a.num_moves)
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
                state=UINT16_ZERO,
                x=INT32_ZERO,
                y=INT32_ZERO,
                orientation=Automaton.Orientation(randrange(4)),
            )
            offspring.append(child)
        self.automata[len(self.automata) // 2 :] = offspring
        for a in self.automata:
            a.reset_stats()
            a.fitness = 0.0


if __name__ == "__main__":
    import signal

    FPS = 10
    TICKS_PER_EVOLVE = 100

    population = Population(size=100)
    rng = default_rng(42)
    maze = population.maze
    assert isinstance(maze, Maze)
    renderer = MazeRenderer(maze)
    tick_count = [0]

    def _simulation_step():
        population.tick()
        tick_count[0] += 1

        if tick_count[0] % TICKS_PER_EVOLVE == 0:
            population.evolve()
        renderer.render(population.automata)

    _timer = renderer.fig.canvas.new_timer(interval=max(1, 1000 // FPS))
    _timer.add_callback(_simulation_step)
    _timer.start()

    def _sigint_handler(sig, frame):
        _timer.stop()
        plt.close("all")

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        plt.show()
    finally:
        renderer.close()
