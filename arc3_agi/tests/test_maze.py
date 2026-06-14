"""Tests for the generate_maze function and maze class."""

import sys
from collections import deque
from unittest.mock import MagicMock

import numpy as np
import pytest

# terry2 fails to fully initialise in test context (missing arcengine deps);
# mock the submodule so generate_maze (which has no terry2 dependency) can import.
_mock = MagicMock()
_mock.Environment2DGrid = object  # plain base class so maze() class body is valid
sys.modules.setdefault("arc3_agi.terry2", _mock)

from arc3_agi.maze import generate_maze  # noqa: E402


def flood_fill_free(wall: np.ndarray) -> int:
    """Return the number of free cells reachable from any free cell via 4-connectivity."""
    side = wall.shape[0]
    free = np.argwhere(wall == 0)
    if len(free) == 0:
        return 0
    start = tuple(free[0])
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < side and 0 <= nc < side and wall[nr, nc] == 0:
                pos = (nr, nc)
                if pos not in visited:
                    visited.add(pos)
                    queue.append(pos)
    return len(visited)


@pytest.mark.parametrize("bits", [4, 5, 6])
def test_output_shape_and_dtype(bits: int) -> None:
    side = 2**bits
    wall, goal = generate_maze(bits)
    assert wall.shape == (side, side)
    assert goal.shape == (side, side)
    assert wall.dtype == np.uint8
    assert goal.dtype == np.uint8


@pytest.mark.parametrize("bits", [4, 5, 6])
def test_outer_border_is_wall(bits: int) -> None:
    wall, _ = generate_maze(bits)
    side = 2**bits
    assert np.all(wall[0, :] == 1), "top row must be wall"
    assert np.all(wall[-1, :] == 1), "bottom row must be wall"
    assert np.all(wall[:, 0] == 1), "left column must be wall"
    assert np.all(wall[:, -1] == 1), "right column must be wall"


@pytest.mark.parametrize("bits", [4, 5, 6])
def test_values_are_binary(bits: int) -> None:
    wall, goal = generate_maze(bits)
    assert set(np.unique(wall)).issubset({0, 1})
    assert set(np.unique(goal)).issubset({0, 1})


@pytest.mark.parametrize("bits", [4, 5, 6])
def test_connectivity(bits: int) -> None:
    """All free cells must form one connected region."""
    wall, _ = generate_maze(bits)
    total_free = int(np.sum(wall == 0))
    assert total_free > 0
    reachable = flood_fill_free(wall)
    assert (
        reachable == total_free
    ), f"bits={bits}: only {reachable}/{total_free} free cells are connected"


@pytest.mark.parametrize("bits", [4, 5, 6])
def test_exactly_one_goal(bits: int) -> None:
    _, goal = generate_maze(bits)
    assert int(np.sum(goal == 1)) == 1


@pytest.mark.parametrize("bits", [4, 5, 6])
def test_goal_on_free_cell(bits: int) -> None:
    wall, goal = generate_maze(bits)
    gr, gc = np.argwhere(goal == 1)[0]
    assert wall[gr, gc] == 0, "goal must be placed on a free cell"


def test_invalid_bits_raises() -> None:
    with pytest.raises(ValueError):
        generate_maze(3)


def test_reproducibility() -> None:
    wall1, goal1 = generate_maze(4, seed=42)
    wall2, goal2 = generate_maze(4, seed=42)
    np.testing.assert_array_equal(wall1, wall2)
    np.testing.assert_array_equal(goal1, goal2)


def test_different_seeds_differ() -> None:
    wall1, _ = generate_maze(4, seed=1)
    wall2, _ = generate_maze(4, seed=2)
    assert not np.array_equal(wall1, wall2)


@pytest.mark.parametrize("bits", [4, 5, 6])
def test_no_loops_tree_structure(bits: int) -> None:
    """For a perfect maze: free cells + passages form a spanning tree.

    A spanning tree on N nodes has exactly N-1 edges.
    Count interior edges (pairs of adjacent free cells).
    Number of free cells = N.  Edges = N - 1.
    """
    wall, _ = generate_maze(bits)
    side = 2**bits
    free_count = int(np.sum(wall == 0))
    # Count horizontal and vertical adjacencies between free cells.
    h_edges = int(np.sum((wall[1:-1, 1:-1] == 0) & (wall[1:-1, 2:] == 0)))
    v_edges = int(np.sum((wall[1:-1, 1:-1] == 0) & (wall[2:, 1:-1] == 0)))
    edges = h_edges + v_edges
    assert (
        edges == free_count - 1
    ), f"bits={bits}: expected {free_count - 1} edges for a tree, got {edges}"
