from enum import IntEnum
from operator import is_
from signal import SIGINT, signal
from typing import Optional

import matplotlib
from numpy import (
    argwhere,
    array,
    ones,
    uint8,
    zeros,
)
from numpy.random import default_rng
from numpy.typing import NDArray

from arc3_agi.automaton import ActionStatus, AutomatonISBase
from arc3_agi.environment import LayeredStaticBoolean2DGrid, StaticBoolean2DGrid
from arc3_agi.genetic_code import GeneticCode, GeneticCodeDict
from arc3_agi.population import Population

matplotlib.use("webagg")
import matplotlib.pyplot as plt


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
        1 = goal cell, 0 = non-goal.  Exactly one cell is the goal,
        and it is always a free cell (i.e. the corresponding cell in `wall` is 0).

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

    def cell_to_grid(cr: int, cc: int) -> tuple[int, int]:
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


class Maze(LayeredStaticBoolean2DGrid):
    """A maze environment represented as a layered 2D grid.

    The maze consists of walls and a goal, represented as two layers in the grid.
    """

    class LKEYS(IntEnum):
        WALL = 0
        GOAL = 1

    Orientation = LayeredStaticBoolean2DGrid.Orientation

    def __init__(
        self, name: str, side_length_bits: int, seed: Optional[int] = None
    ) -> None:
        """Initializes the maze environment.

        Args:
            name: The name of the maze environment.
            side_length_bits: The number of bits to determine the side length of the maze (2^side_length_bits).
            seed: An optional integer seed for reproducible maze generation.
        """
        wall, goal = generate_maze(side_length_bits, seed)
        super().__init__(
            name=name,
            grid=[
                [int(wall[i, j]) for j in range(wall.shape[1])]
                for i in range(wall.shape[0])
            ],
            num_layers=1,
        )
        self.add_layer(
            StaticBoolean2DGrid(
                name=f"{name}_goal",
                grid=[
                    [int(goal[i, j]) for j in range(goal.shape[1])]
                    for i in range(goal.shape[0])
                ],
            )
        )
        self.width = wall.shape[1]
        self.height = wall.shape[0]
        self.wall_grid = self.layers[self.LKEYS.WALL].get()
        self.goal_grid = self.layers[self.LKEYS.GOAL].get()
        self.wall_layer = self.layers[self.LKEYS.WALL]
        self.goal_layer = self.layers[self.LKEYS.GOAL]
        self.rng = default_rng(seed)
        self.free = [
            (x, y)
            for y, row in enumerate(self.wall_grid)
            for x, v in enumerate(row)
            if not v
        ]

    def get_local(self, coords: list[int], **kwargs) -> bytes:
        """Returns the local environment stimulus for the given coordinates.
        NOTE: This only returns the wall layer.
        """
        return self.wall_layer.get_local(coords, **kwargs)

    def is_goal(self, x: int, y: int) -> bool:
        """Checks if the cell at (x, y) is the goal."""
        return self.goal_grid[y][x]

    def is_wall(self, x: int, y: int) -> bool:
        """Checks if the cell at (x, y) is a wall."""
        return self.wall_grid[y][x]

    def random_free_cell(self, seed: Optional[int] = None) -> tuple[int, int]:
        """Returns a random free (non-wall) cell as (x, y)."""
        idx = int(self.rng.integers(0, len(self.free)))
        return self.free[idx]


class MazeAutomaton(AutomatonISBase):
    """Represents the automaton for Terry's world."""

    def __init__(self, **kwargs) -> None:
        """Initialize the automaton.

        Expects kwargs:
            name: Optional name for the automaton.
            genetic_code: Optional genetic code for the automaton. If not provided,
                a default empty code is used.
            x: Initial x-coordinate (column) of the automaton in the maze. Default is 0.
            y: Initial y-coordinate (row) of the automaton in the maze. Default is 0.
            orientation: Initial orientation of the automaton. Should be an instance
                of Maze.Orientation. Default is Maze.Orientation.UP.
            environment: The Maze environment instance that the automaton will interact with.
                This is required.
        """
        super().__init__(
            name=kwargs.get("name", "Terry-2"),
            genetic_code=kwargs.get("genetic_code", None),
            env_bits=9,
            state_bits=8,
            resp_bits=2,
            environment=kwargs.get("environment"),
        )
        if self.genetic_code is None:
            self.genetic_code = GeneticCodeDict(
                {}, resp_bits=(self.state_bytes + self.resp_bytes) << 3
            )
        assert isinstance(
            self.environment, Maze
        ), "MazeAutomaton requires a Maze environment."
        fx, fy = self.environment.random_free_cell()
        self.coords = [
            kwargs.get("x", fx),
            kwargs.get("y", fy),
            kwargs.get("orientation", Maze.Orientation.UP).value,
        ]

        # Since automata do not interact in anyway, even through the environment
        # they each have a separate energy grid for tracking coverage and
        # encouraging exploration.
        # If an automaton moves into a cell, it gets energy for that cell once
        # and then that cell is "depleted" for that automaton. This encourages
        # exploration of new cells rather than camping in one place or going back
        # and forth between a few cells.
        # NOTE: The automaton cannot 'see' the energy grid; it is only used for
        # shaping the fitness function so that the automaton is incentivized to
        # explore the maze and find the goal.
        self.energy: int = 10  # Initial energy for the automaton; can be tuned.
        self.energy_grid: list[list[bool]] = [
            [True for _ in range(self.environment.width)]
            for _ in range(self.environment.height)
        ]
        self.fitness = 0.0

    @property
    def x(self) -> int:
        return self.coords[0]

    @property
    def y(self) -> int:
        return self.coords[1]

    @property
    def orientation(self) -> Maze.Orientation:
        return Maze.Orientation(self.coords[2])

    def attempt_action(self, action: bytes) -> ActionStatus:
        """Perform the given action."""
        action_int = (action[0] if action else 0) & ((1 << self.resp_bits) - 1)
        self.last_action = action_int
        match action_int:
            case 0:  # Move forward
                move = Maze.orientation_moves[Maze.Orientation(self.coords[2])]
                dx = move[0] + self.coords[0]
                dy = move[1] + self.coords[1]

                # Check for wall collision and bounds
                assert isinstance(
                    self.environment, Maze
                ), "MazeAutomaton requires a Maze environment."
                if self.environment.is_wall(dx, dy):
                    # NOTE: The environment is bordered by walls, so out-of-bounds
                    # is also a wall collision.
                    return ActionStatus.FAILED

                # If it is free space move
                x = self.coords[0] + move[0]
                y = self.coords[1] + move[1]

                # See if energy is there
                energy = self.energy_grid[y][x]
                self.fitness += energy
                self.energy += energy * 2  # Gain energy for moving into a new cell.
                self.energy_grid[y][x] = False

                # NOTE: Ignoring a goal for now.
                # Is it the goal?
                # if self.environment.is_goal(x, y):
                #    self.fitness += 100.0  # Big fitness boost for reaching the goal.
                #    self.energy += 50  # Bonus energy for reaching the goal

                self.coords[0] = x
                self.coords[1] = y
                return ActionStatus.SUCCEEDED
            case 1:  # Turn left
                self.coords[2] = (self.coords[2] - 1) & 3
                return ActionStatus.SUCCEEDED
            case 2:  # Turn right
                self.coords[2] = (self.coords[2] + 1) & 3
                return ActionStatus.SUCCEEDED
            case _:
                return ActionStatus.INVALID

    def reset(self) -> None:
        """Resets the automaton's state, energy, and fitness."""
        super().reset()
        assert isinstance(
            self.environment, Maze
        ), "MazeAutomaton requires a Maze environment."
        fx, fy = self.environment.random_free_cell()
        self.coords = [fx, fy, Maze.Orientation.UP.value]
        self.energy = 10  # Reset energy to initial value.
        self.energy_grid = [
            [True for _ in range(self.environment.width)]
            for _ in range(self.environment.height)
        ]
        self.fitness = 0.0

    def tick(self) -> bytes:
        """Perform a tick of the automaton."""
        # super().tick() already updates internal_state and returns only the action bytes.
        action = super().tick()
        self.energy -= 1  # Each tick costs 1 energy; can be tuned.
        self.attempt_action(action)
        return action


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
        wall_layer = array(self.maze.layers[self.maze.LKEYS.WALL].get(), dtype=uint8)
        goal_layer = array(self.maze.layers[self.maze.LKEYS.GOAL].get(), dtype=uint8)
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


class FitnessRenderer:
    """Live histogram showing the fitness distribution of the population."""

    def __init__(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Fitness Distribution")
        self.generation = 0
        self.ax.set_xlabel("Fitness")
        self.ax.set_ylabel("Count")
        self.ax.set_title("Generation 0 – Fitness Distribution")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, fitnesses: list[float]) -> None:
        self.generation += 1
        self.ax.cla()
        self.ax.hist(fitnesses, bins=20, color="steelblue", edgecolor="black")
        mn = min(fitnesses)
        mx = max(fitnesses)
        mean = sum(fitnesses) / len(fitnesses)
        self.ax.axvline(
            mean,
            color="crimson",
            linestyle="--",
            linewidth=1.5,
            label=f"mean={mean:.1f}",
        )
        self.ax.set_xlabel("Fitness")
        self.ax.set_ylabel("Count")
        self.ax.set_title(
            f"Generation {self.generation}  –  min={mn:.1f}  mean={mean:.1f}  max={mx:.1f}"
        )
        self.ax.legend()
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def is_open(self) -> bool:
        return plt.fignum_exists(self.fig.number)

    def close(self) -> None:
        plt.close(self.fig)


if __name__ == "__main__":
    # Example usage: generate and render a maze.
    FPS = 10
    maze = Maze(name="ExampleMaze", side_length_bits=6, seed=42)
    population = Population(size=100, AutomatonClass=MazeAutomaton, environment=maze)
    renderer = MazeRenderer(maze)
    fitness_renderer = FitnessRenderer()

    def _simulation_step():
        import traceback

        try:
            population.tick()
            renderer.render(population.automata[:20])
            if population.tick_count % 50 == 0:  # Evolve every 50 ticks
                fitness_renderer.update(population.evolve())
        except Exception:
            traceback.print_exc()
            _timer.stop()

    _timer = renderer.fig.canvas.new_timer(interval=max(1, 1000 // FPS))
    _timer.add_callback(_simulation_step)
    _timer.start()

    def _sigint_handler(sig, frame):
        _timer.stop()
        plt.close("all")

    signal(SIGINT, _sigint_handler)

    try:
        plt.show()
    finally:
        renderer.close()
        fitness_renderer.close()
