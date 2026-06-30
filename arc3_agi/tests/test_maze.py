"""Tests for the Maze environment, maze generation and the maze automaton."""

import numpy as np
import pytest

from arc3_agi.automaton import ActionStatus
from arc3_agi.genetic_code import GeneticCodeGraph
from arc3_agi.maze import Maze, MazeAutomaton, generate_maze

ORIENTATION_DELTAS = {
    Maze.Orientation.UP: (0, -1),
    Maze.Orientation.RIGHT: (1, 0),
    Maze.Orientation.DOWN: (0, 1),
    Maze.Orientation.LEFT: (-1, 0),
}


# --------------------------------------------------------------------------- #
# generate_maze
# --------------------------------------------------------------------------- #
def test_generate_maze_shape_and_border() -> None:
    wall, goal = generate_maze(4, seed=42)
    assert wall.shape == (16, 16)
    assert goal.shape == (16, 16)
    assert wall[0, :].all()
    assert wall[-1, :].all()
    assert wall[:, 0].all()
    assert wall[:, -1].all()


def test_generate_maze_single_goal_on_free_cell() -> None:
    wall, goal = generate_maze(4, seed=7)
    assert int(goal.sum()) == 1
    ys, xs = np.where(goal == 1)
    assert wall[ys[0], xs[0]] == 0


def test_generate_maze_is_deterministic() -> None:
    wall1, goal1 = generate_maze(4, seed=123)
    wall2, goal2 = generate_maze(4, seed=123)
    assert np.array_equal(wall1, wall2)
    assert np.array_equal(goal1, goal2)


def test_generate_maze_rejects_small_grids() -> None:
    with pytest.raises(ValueError):
        generate_maze(3)


# --------------------------------------------------------------------------- #
# Maze environment
# --------------------------------------------------------------------------- #
def test_maze_dimensions_and_free_cells() -> None:
    maze = Maze("m", 4, seed=1)
    assert maze.width == 16
    assert maze.height == 16
    assert maze.is_wall(0, 0)
    assert len(maze.free) > 0
    fx, fy = maze.random_free_cell()
    assert not maze.is_wall(fx, fy)


def test_maze_get_local_fits_in_nine_bits() -> None:
    maze = Maze("m", 4, seed=1)
    fx, fy = maze.free[0]
    value = maze.get_local([fx, fy, Maze.Orientation.UP.value])
    assert 0 <= value < (1 << 9)


# --------------------------------------------------------------------------- #
# MazeAutomaton
# --------------------------------------------------------------------------- #
def test_turn_actions_rotate_orientation() -> None:
    maze = Maze("m", 4, seed=1)
    fx, fy = maze.free[0]
    auto = MazeAutomaton(environment=maze, x=fx, y=fy, orientation=Maze.Orientation.UP)

    assert auto.attempt_action(1) == ActionStatus.SUCCEEDED  # turn left
    assert auto.coords[2] == Maze.Orientation.LEFT.value

    auto.coords[2] = Maze.Orientation.UP.value
    assert auto.attempt_action(2) == ActionStatus.SUCCEEDED  # turn right
    assert auto.coords[2] == Maze.Orientation.RIGHT.value


def test_move_into_border_wall_fails() -> None:
    maze = Maze("m", 4, seed=1)
    cell = next(((x, y) for (x, y) in maze.free if x == 1), None)
    assert cell is not None, "expected a free cell adjacent to the left border"
    x, y = cell
    auto = MazeAutomaton(environment=maze, x=x, y=y, orientation=Maze.Orientation.LEFT)
    assert auto.attempt_action(0) == ActionStatus.FAILED
    assert auto.coords[0] == x
    assert auto.coords[1] == y


def test_move_forward_into_free_cell_succeeds_and_rewards() -> None:
    maze = Maze("m", 4, seed=1)
    for x, y in maze.free:
        for orientation, (dx, dy) in ORIENTATION_DELTAS.items():
            nx, ny = x + dx, y + dy
            if not maze.is_wall(nx, ny):
                auto = MazeAutomaton(
                    environment=maze, x=x, y=y, orientation=orientation
                )
                fitness_before = auto.fitness
                assert auto.attempt_action(0) == ActionStatus.SUCCEEDED
                assert auto.coords[0] == nx
                assert auto.coords[1] == ny
                assert auto.fitness > fitness_before
                return
    pytest.fail("no pair of adjacent free cells found in the maze")


def test_graph_code_drives_tick_within_response_mask() -> None:
    maze = Maze("m", 4, seed=1)
    code = GeneticCodeGraph.random(input_bits=14, resp_bits=7, num_nodes=24, seed=5)
    auto = MazeAutomaton(environment=maze, genetic_code=code)
    for _ in range(20):
        response = auto.tick()
        assert 0 <= response < (1 << 2)
