from enum import IntEnum
from operator import is_
from signal import SIGINT, signal
from typing import Callable, Optional

import matplotlib
import numpy as np
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
from arc3_agi.checkpoint import CheckpointConfig
from arc3_agi.environment import LayeredStaticBoolean2DGrid, StaticBoolean2DGrid
from arc3_agi.fingerprint import FingerprintConfig
from arc3_agi.genetic_code import GeneticCodeDict, GeneticCodeGraph
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
            radius=1,
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

    def get_local(self, coords: list[int], **kwargs) -> int:
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
            state_bits=4,
            resp_bits=2,
            environment=kwargs.get("environment"),
            fingerprint_config=kwargs.get("fingerprint_config", None),
            seed=kwargs.get("seed", None),
        )
        if self.genetic_code is None:
            self.genetic_code = GeneticCodeDict(
                {},
                seed=self.rng.randint(0, 2**32 - 1),
                resp_bits=self.state_bits + self.resp_bits,
            )
        assert isinstance(
            self.environment, Maze
        ), "MazeAutomaton requires a Maze environment."
        # Use self.rng (seeded from Population) to decouple starting position
        # from the shared Maze.rng state.
        free = self.environment.free
        fx, fy = free[self.rng.randrange(len(free))]
        self.coords = [
            kwargs.get("x", fx),
            kwargs.get("y", fy),
            kwargs.get("orientation", Maze.Orientation(self.rng.randint(0, 3))).value,
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
        self.energy: int = 15  # Initial energy budget; MUST be << TICKS_PER_GEN.
        # A zero-gain automaton (invalid-looper) loses 1 energy/tick and gains
        # nothing, so with energy=10 it dies after 10 ticks and wastes none of
        # the remaining 90.  A good explorer gains +2 per new cell visited, so
        # visiting just 5 new cells in the first 10 ticks keeps it alive for
        # the full generation.
        self._grid_width: int = self.environment.width
        self.energy_grid: bytearray = bytearray(
            b"\x01" * (self.environment.width * self.environment.height)
        )
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

    @property
    def is_active(self) -> bool:
        """Return False once energy is exhausted.

        Population.tick() checks this before each tick so that automata that
        have run out of energy (typically invalid-loopers or chronic wall-
        bangers) stop consuming CPU for the remainder of the generation.
        The automaton remains in the population list and its accumulated
        (possibly negative) fitness is used normally at breeding time.
        """
        return self.energy > 0

    def _gen_mkvfn(self) -> Callable[[], int]:
        """Return a function that returns the automaton's current state and
        environment stimulus as a single integer, suitable for use as a key
        in the genetic code graph.
        """

        def mkvfn() -> int:
            # A zero internal state is only allowed on reset().
            # It is an indication that the automaton has not yet been ticked
            # and is not yet in a valid state.
            internal_state = self.rng.randint(1, (1 << self.state_bits) - 1)
            action = self.rng.randint(0, 2)
            return (internal_state << self.env_bits) | action

        return mkvfn

    def attempt_action(self, action: int) -> ActionStatus:
        """Perform the given action."""
        action_int = action & self.resp_mask
        self.last_action = action_int
        match action_int:
            case 0:  # Move forward
                move = Maze.orientation_moves[self.coords[2]]
                dx = move[0] + self.coords[0]
                dy = move[1] + self.coords[1]

                # Check for wall collision and bounds
                assert isinstance(
                    self.environment, Maze
                ), "MazeAutomaton requires a Maze environment."
                if self.environment.is_wall(dx, dy):
                    # NOTE: The environment is bordered by walls, so out-of-bounds
                    # is also a wall collision.
                    self.fitness -= 0.05  # B: small penalty for wasted move into wall
                    return ActionStatus.FAILED

                # If it is free space move
                x = self.coords[0] + move[0]
                y = self.coords[1] + move[1]

                # See if energy is there
                idx = y * self._grid_width + x
                energy = self.energy_grid[idx]
                self.fitness += energy + 0.1  # Tiny boost for any forward move.
                self.energy += energy * 2  # Gain energy for moving into a new cell.
                self.energy_grid[idx] = 0

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
                self.fitness -= (
                    0.1  # B: larger penalty for invalid (meaningless) action
                )
                return ActionStatus.INVALID

    def reset(self) -> None:
        """Resets the automaton's state, energy, and fitness."""
        super().reset()
        assert isinstance(
            self.environment, Maze
        ), "MazeAutomaton requires a Maze environment."
        # Use the automaton's own rng so each automaton's starting position is
        # independent and derived purely from its own seed — not coupled to
        # the shared Maze.rng state.
        free = self.environment.free
        fx, fy = free[self.rng.randrange(len(free))]
        random_orientation = Maze.Orientation(self.rng.randint(0, 3))
        self.coords = [fx, fy, random_orientation.value]
        self.energy = 15  # Reset energy to initial value.
        self.energy_grid = bytearray(
            b"\x01" * (self.environment.width * self.environment.height)
        )
        self.fitness = 0.0

    # ------------------------------------------------------------------
    # Checkpoint interface
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["automaton"]["energy"] = self.energy
        return d

    def to_arrays(self) -> dict:
        arrays = super().to_arrays()
        arrays["energy_grid"] = np.frombuffer(self.energy_grid, dtype=np.uint8).copy()
        return arrays

    @classmethod
    def from_dict(cls, d, arrays, **kwargs):
        inst = super().from_dict(d, arrays, **kwargs)
        assert isinstance(
            inst, MazeAutomaton
        ), "from_dict did not return a MazeAutomaton instance."
        inst.energy = d["automaton"]["energy"]
        inst.energy_grid = bytearray(arrays["energy_grid"].tobytes())
        assert isinstance(inst.environment, Maze)
        inst._grid_width = inst.environment.width
        return inst

    def tick(self) -> int:
        """Perform a tick of the automaton."""
        # super().tick() already updates internal_state and returns only the action bytes.
        action = super().tick()
        self.energy -= 1  # Each tick costs 1 energy; can be tuned.
        self.attempt_action(action)
        return action


def _moving_average(data: list[float], window: int) -> list[float]:
    """Return a simple moving average of *data* with the given *window*.

    Positions where fewer than *window* values are available are filled with
    ``float("nan")`` so they are invisible on a matplotlib plot.
    """
    result: list[float] = []
    for i in range(len(data)):
        if i + 1 < window:
            result.append(float("nan"))
        else:
            result.append(sum(data[i + 1 - window : i + 1]) / window)
    return result


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


class FitnessHistoryRenderer:
    """Live line chart showing mean and max fitness per generation."""

    def __init__(self, window: int = 10) -> None:
        self._window = window
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Fitness History")
        self._means: list[float] = []
        self._maxes: list[float] = []
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")
        self.ax.set_title("Fitness History")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, fitnesses: list[float]) -> None:
        self._means.append(sum(fitnesses) / len(fitnesses))
        self._maxes.append(max(fitnesses))
        gens = list(range(1, len(self._means) + 1))
        ma_means = _moving_average(self._means, self._window)
        ma_maxes = _moving_average(self._maxes, self._window)
        self.ax.cla()
        self.ax.plot(gens, self._means, color="crimson", linewidth=1.5, label="mean")
        self.ax.plot(
            gens,
            ma_means,
            color="crimson",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            label=f"mean MA-{self._window}",
        )
        self.ax.plot(gens, self._maxes, color="steelblue", linewidth=1.5, label="max")
        self.ax.plot(
            gens,
            ma_maxes,
            color="steelblue",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            label=f"max MA-{self._window}",
        )
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Fitness")
        self.ax.set_title("Fitness History")
        self.ax.legend()
        self.fig.tight_layout()
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


class GenerationsPerSecondRenderer:
    """Live line chart showing generations per second over time, with a moving average trend line."""

    def __init__(self, window: int = 10) -> None:
        self._window = window
        self._rates: list[float] = []
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Generations per Second")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Generations / s")
        self.ax.set_title("Generations per Second")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, duration_s: float | None) -> None:
        if duration_s is None or duration_s <= 0.0:
            return
        self._rates.append(1.0 / duration_s)
        gens = list(range(1, len(self._rates) + 1))
        ma = _moving_average(self._rates, self._window)
        self.ax.cla()
        self.ax.plot(
            gens, self._rates, color="steelblue", linewidth=1.5, label="gens/s"
        )
        self.ax.plot(
            gens,
            ma,
            color="steelblue",
            linewidth=1.5,
            linestyle="--",
            alpha=0.7,
            label=f"MA-{self._window}",
        )
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("Generations / s")
        self.ax.set_title("Generations per Second")
        self.ax.legend()
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def is_open(self) -> bool:
        return plt.fignum_exists(self.fig.number)

    def close(self) -> None:
        plt.close(self.fig)


class FitnessRateRenderer:
    """Live chart showing the per-generation rate of change of max and mean fitness, with moving average trend lines."""

    def __init__(self, window: int = 100) -> None:
        self._window = window
        self._prev_max: float | None = None
        self._prev_mean: float | None = None
        self._delta_maxes: list[float] = []
        self._delta_means: list[float] = []
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Fitness Rate of Change")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("\u0394 Fitness")
        self.ax.set_title("Fitness Rate of Change")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, fitnesses: list[float]) -> None:
        cur_max = max(fitnesses)
        cur_mean = sum(fitnesses) / len(fitnesses)
        if self._prev_max is None:
            self._delta_maxes.append(float("nan"))
            self._delta_means.append(float("nan"))
        else:
            self._delta_maxes.append(cur_max - self._prev_max)
            self._delta_means.append(cur_mean - self._prev_mean)  # type: ignore[operator]
        self._prev_max = cur_max
        self._prev_mean = cur_mean

        gens = list(range(1, len(self._delta_maxes) + 1))
        ma_max = _moving_average(self._delta_maxes, self._window)
        ma_mean = _moving_average(self._delta_means, self._window)
        self.ax.cla()
        self.ax.plot(
            gens,
            ma_max,
            color="steelblue",
            linewidth=1.5,
            label=f"\u0394 max MA-{self._window}",
        )
        self.ax.plot(
            gens,
            ma_mean,
            color="crimson",
            linewidth=1.5,
            label=f"\u0394 mean MA-{self._window}",
        )
        self.ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        self.ax.set_xlabel("Generation")
        self.ax.set_ylabel("\u0394 Fitness")
        self.ax.set_title("Fitness Rate of Change")
        self.ax.legend()
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def is_open(self) -> bool:
        return plt.fignum_exists(self.fig.number)

    def close(self) -> None:
        plt.close(self.fig)


class FingerprintClusterRenderer:
    """Live 2-D PCA scatter plot of selection-fingerprint bit vectors.

    Each point represents one automaton.  Fingerprints are unpacked into
    binary row vectors, mean-centred, and projected onto the first two
    principal components via SVD.  Points are coloured by current fitness
    so that the relationship between compatibility clusters and performance
    is immediately visible.

    When all fingerprints are identical (zero variance) the plot shows all
    points at the origin, which is still informative (no diversity yet).
    """

    def __init__(self) -> None:
        self._generation: int = 0
        self._cbar = None
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Fingerprint Clusters")
        self.ax.set_xlabel("PC 1")
        self.ax.set_ylabel("PC 2")
        self.ax.set_title("Generation 0 \u2013 Fingerprint Clusters")
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, automata: list, fitnesses: list[float]) -> None:
        """Replot using the current fingerprints and pre-reset fitnesses.

        ``fitnesses`` must be the list returned by ``Population.evolve()``
        (captured before automata are reset) and must be the same length as
        ``automata``, in the same order.

        Each bubble represents one *unique* fingerprint value.  Bubble area is
        proportional to the number of automata sharing that fingerprint (so
        a cluster of 60 automata is visually distinguishable from a singleton),
        and colour encodes the mean fitness of that group.  The title shows the
        diversity ratio: unique fingerprints present vs. the total number of
        possible fingerprints (2^bits).
        """
        fps = [a.fingerprint for a in automata if a.fingerprint is not None]
        if not fps:
            return
        bits = fps[0].bits
        n = len(fps)

        # Unpack integer fingerprints into a (n, bits) binary float matrix.
        values = np.array([fp.value for fp in fps], dtype=np.int64)
        fitnesses_arr = np.array(
            [f for a, f in zip(automata, fitnesses) if a.fingerprint is not None],
            dtype=np.float32,
        )
        bit_positions = np.arange(bits, dtype=np.int64)
        X = ((values[:, None] >> bit_positions[None, :]) & 1).astype(np.float32)

        # PCA via SVD on the mean-centred matrix.
        X_c = X - X.mean(axis=0)
        if X_c.any() and n >= 2:
            try:
                _, _S, Vt = np.linalg.svd(X_c, full_matrices=False)
                n_components = min(2, Vt.shape[0])
                coords_2d = X_c @ Vt[:n_components].T  # shape (n, ≤2)
                if coords_2d.shape[1] < 2:
                    coords_2d = np.column_stack(
                        [coords_2d, np.zeros(n, dtype=np.float32)]
                    )
            except np.linalg.LinAlgError:
                coords_2d = np.zeros((n, 2), dtype=np.float32)
        else:
            coords_2d = np.zeros((n, 2), dtype=np.float32)

        # --- Aggregate by unique fingerprint value ---
        # All automata sharing the same integer fingerprint also share the same
        # PCA coordinate (since PCA is a deterministic linear map of the bits),
        # so we can safely average coords_2d per group.
        unique_vals, inverse = np.unique(values, return_inverse=True)
        n_unique = len(unique_vals)
        agg_x = np.zeros(n_unique, dtype=np.float32)
        agg_y = np.zeros(n_unique, dtype=np.float32)
        agg_fit = np.zeros(n_unique, dtype=np.float32)
        agg_count = np.zeros(n_unique, dtype=np.int32)
        np.add.at(agg_x, inverse, coords_2d[:, 0])
        np.add.at(agg_y, inverse, coords_2d[:, 1])
        np.add.at(agg_fit, inverse, fitnesses_arr)
        np.add.at(agg_count, inverse, 1)
        agg_x /= agg_count
        agg_y /= agg_count
        agg_fit /= agg_count

        # Bubble area ∝ count (min 30 pt², max 600 pt²).
        max_count = int(agg_count.max())
        sizes = 30 + 570 * (agg_count / max(max_count, 1))

        self._generation += 1
        f_min = float(fitnesses_arr.min())
        f_max = float(fitnesses_arr.max())
        f_range = f_max - f_min if f_max > f_min else 1.0
        n_possible = 1 << bits

        # Remove old colourbar before clearing axes.
        if self._cbar is not None:
            self._cbar.remove()
            self._cbar = None

        self.ax.cla()
        sc = self.ax.scatter(
            agg_x,
            agg_y,
            c=agg_fit,
            s=sizes,
            cmap="plasma",
            alpha=0.85,
            vmin=f_min,
            vmax=f_min + f_range,
            linewidths=0.5,
            edgecolors="gray",
        )
        # Annotate each bubble with its population count.
        for i in range(n_unique):
            self.ax.annotate(
                str(agg_count[i]),
                (float(agg_x[i]), float(agg_y[i])),
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )
        self._cbar = self.fig.colorbar(sc, ax=self.ax, label="Mean Fitness")
        self.ax.set_xlabel("PC 1")
        self.ax.set_ylabel("PC 2")
        self.ax.set_title(
            f"Generation {self._generation} \u2013 Fingerprint Clusters\n"
            f"{n_unique}/{n_possible} unique fingerprints  (area \u221d count)"
        )
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def is_open(self) -> bool:
        return plt.fignum_exists(self.fig.number)

    def close(self) -> None:
        plt.close(self.fig)


if __name__ == "__main__":
    import traceback

    # Example usage: generate and render a maze.
    FPS = 10
    TICKS_PER_RESTART = 100  # Ticks simulated per restart before evolving.
    WATCH_EVERY = 100  # Animate the maze every Nth generation; others run headless.
    maze = Maze(name="ExampleMaze", side_length_bits=6, seed=42)
    population = Population(
        size=100,
        AutomatonClass=MazeAutomaton,
        environment=maze,
        checkpoint_config=CheckpointConfig(generation_interval=WATCH_EVERY),
        fingerprint_config=FingerprintConfig(bits=4, tournament_k=4),
    )
    renderer = MazeRenderer(maze)
    fitness_renderer = FitnessRenderer()
    fitness_history_renderer = FitnessHistoryRenderer()
    gens_per_sec_renderer = GenerationsPerSecondRenderer()
    fitness_rate_renderer = FitnessRateRenderer()
    fingerprint_cluster_renderer = FingerprintClusterRenderer()

    # The maze renderer is by far the most expensive part of a tick, so we only
    # animate it once every WATCH_EVERY generations. The intervening generations
    # are simulated "headless" (no per-tick maze rendering) as fast as possible.
    # The fitness charts still update on every generation.
    _state = {"generation": 0, "tick": 0, "watching": True}

    def _finish_generation():
        fitnesses = population.evolve()
        fitness_renderer.update(fitnesses)
        fitness_history_renderer.update(fitnesses)
        gens_per_sec_renderer.update(population.fitness_history[-1].get("duration_s"))
        fitness_rate_renderer.update(fitnesses)
        fingerprint_cluster_renderer.update(population.automata, fitnesses)
        _state["generation"] += 1
        _state["tick"] = 0
        _state["watching"] = _state["generation"] % WATCH_EVERY == 0

    def _simulation_step():
        try:
            if _state["watching"]:
                # Animate one tick per timer callback so behaviour is visible.
                population.tick()
                renderer.render(population.automata[:20])
                _state["tick"] += 1
                if _state["tick"] >= TICKS_PER_RESTART:
                    _finish_generation()
            else:
                # Burn through whole generations headless until the next watched
                # generation, then hand back to the animation path above.
                while not _state["watching"]:
                    for _ in range(TICKS_PER_RESTART):
                        population.tick()
                    _finish_generation()
                renderer.render(population.automata[:20])
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
        fitness_history_renderer.close()
        gens_per_sec_renderer.close()
        fitness_rate_renderer.close()
        fingerprint_cluster_renderer.close()
