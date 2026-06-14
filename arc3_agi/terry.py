"""Terry module."""

from __future__ import annotations

from enum import IntEnum
from functools import lru_cache
from random import randrange

from numpy import arange, array, empty, uint8, zeros
from numpy.typing import NDArray

# Constants
UINT8_ZERO = uint8(0)
UINT8_ONE = uint8(1)


class Orientation(IntEnum):
    """Represents the possible orientations for the automaton."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class AutomatonAction(IntEnum):
    """Represents the possible actions for the automaton."""

    CFRAME_INCREMENT = 0
    CFRAME_DECREMENT = 1
    MOVE_FORWARD = 2
    TURN_LEFT = 3
    TURN_RIGHT = 4
    TELEPORT = 5
    ZOOM_IN = 6
    ZOOM_OUT = 7
    DO_NOTHING = 8


class FrameBase:
    """Represents a single frame of the game state."""

    def __init_subclass__(cls, size: int = 64, nbits: int = 8, **kwargs) -> None:
        """Initialize a Frame subclass.

        Args:
            size: The size of the grid (size x size). Default is 64.
            nbits: The number of bits used to represent each cell in the grid. Default is 8,
                allowing for values from 0 to 255.
        """
        super().__init_subclass__(**kwargs)
        cls.size = size
        cls.nbits = nbits
        cls.wrap_mask = (1 << nbits) - 1

        # Strict sanity checks to ensure the grid is valid
        assert size.bit_count() == 1, "Size must be a power of 2"
        assert size >= 4, "Size must be >= 4"
        assert size <= 1024, "Size must be <= 1024 to prevent excessive memory usage"
        assert nbits > 0, "Number of bits must be positive"
        assert nbits <= 8, "Number of bits must be 8 or less to fit in uint8"

    def __init__(self) -> None:
        """Initialize the grid for the frame."""
        self.grid: NDArray[uint8] = empty((self.size, self.size), dtype=uint8)

    def set(self, x: int, y: int, value: int) -> None:
        """Set the value of a cell in the grid.

        Args:
            x: The x-coordinate of the cell (0 <= x < size).
            y: The y-coordinate of the cell (0 <= y < size).
            value: The value to set in the cell (0 <= value < 2^nbits).

        Raises:
            ValueError: If the coordinates are out of bounds or if the value is invalid.
        """
        if not (0 <= x < self.size):
            raise ValueError(f"x coordinate {x} is out of bounds")
        if not (0 <= y < self.size):
            raise ValueError(f"y coordinate {y} is out of bounds")
        if not (0 <= value < (1 << self.nbits)):
            raise ValueError(
                f"value {value} is out of valid range for nbits={self.nbits}"
            )

        self.grid[y, x] = uint8(value)

    def get(self, x: int, y: int, radius: int = 0) -> NDArray[uint8]:
        """Get the value of a cell in the grid.

        Args:
            x: The x-coordinate of the cell (0 <= x < size).
            y: The y-coordinate of the cell (0 <= y < size).
            radius: The radius around the cell to consider (default is 0, meaning only the cell itself).

        Returns:
            The value of the cell or the aggregated value within the radius.

        Raises:
            ValueError: If the coordinates are out of bounds or if the radius is invalid.
        """
        if not (0 <= x < self.size):
            raise ValueError(f"x coordinate {x} is out of bounds")
        if not (0 <= y < self.size):
            raise ValueError(f"y coordinate {y} is out of bounds")
        if not (0 <= radius < self.size):
            raise ValueError(f"radius {radius} is out of valid range")

        x_min = x - radius
        x_max = (x + radius + 1) & self.wrap_mask
        y_min = y - radius
        y_max = (y + radius + 1) & self.wrap_mask
        return self.grid[y_min:y_max, x_min:x_max]


class Frame64x4(FrameBase, size=64, nbits=4):
    """Represents a single frame of the game state, inheriting from FrameBase."""


class Frame64x1(FrameBase, size=64, nbits=1):
    """Represents a single frame of the game state, inheriting from FrameBase."""


class Frame32x1(FrameBase, size=32, nbits=1):
    """Represents a single frame of the game state, inheriting from FrameBase."""


class Frame16x1(FrameBase, size=16, nbits=1):
    """Represents a single frame of the game state, inheriting from FrameBase."""


class Frame8x1(FrameBase, size=8, nbits=1):
    """Represents a single frame of the game state, inheriting from FrameBase."""


class Frame4x1(FrameBase, size=4, nbits=1):
    """Represents a single frame of the game state, inheriting from FrameBase."""


class AutomatonBase:
    """Base class for automata in the game."""

    def __init_subclass__(cls, radius: int = 1, isize: int = 8, **kwargs) -> None:
        """Initialize an Automaton subclass."""
        super().__init_subclass__(**kwargs)
        cls.radius = radius
        cls.isize = isize
        cls._orientation_indices = cls._generate_orientation_indices()

    @classmethod
    def _generate_orientation_indices(cls) -> NDArray[uint8]:
        """An Automaton only sees the local frame in its current orientation. This method generates
        a mapping of grid coordinates to indices for all orientations as a reference.

        Using the example of a 3x3 grid the FrameBase grid coordinates are
            (x-1, y-1) (x, y-1) (x+1, y-1)
            (x-1, y)   (x, y)   (x+1, y)
            (x-1, y+1) (x, y+1) (x+1, y+1)
        In the local frame this is which corresponds to "UP" orientation:
            (0, 0) (1, 0) (2, 0)
            (0, 1) (1, 1) (2, 1)
            (0, 2) (1, 2) (2, 2)
        Converting to numpy [row, column] indexing
            row [0, 0, 0, 1, 1, 1, 2, 2, 2]
            col [0, 1, 2, 0, 1, 2, 0, 1, 2]
        The numpy [row, column] indexing for "LEFT" orientation is effectively a 90 degree clockwise
        rotation of the "UP" orientation, so the indexing becomes:
            row [2, 1, 0, 2, 1, 0, 2, 1, 0]
            col [0, 0, 0, 1, 1, 1, 2, 2, 2]
        "DOWN" orientation is a 180 degree rotation of "UP":
            row [2, 2, 2, 1, 1, 1, 0, 0, 0]
            col [2, 1, 0, 2, 1, 0, 2, 1, 0]
        "RIGHT" orientation is a 270 degree rotation of "UP":
            row [0, 1, 2, 0, 1, 2, 0, 1, 2]
            col [2, 2, 2, 1, 1, 1, 0, 0, 0]

        The returned array has shape (4, 2, (radius*2 + 1)**2) where the first dimension corresponds to the
        orientation and the second dimension corresponds row, column and the third dimension
        corresponds to the flattened indices.
        """
        size = cls.radius * 2 + 1
        indices = empty((4, 2, size * size), dtype=uint8)
        for orientation in Orientation:
            match orientation:
                case Orientation.UP:
                    row = array(
                        (r for r in range(size) for _ in range(size)), dtype=uint8
                    )
                    col = array(
                        (c for _ in range(size) for c in range(size)), dtype=uint8
                    )
                case Orientation.LEFT:
                    row = array(
                        (r for _ in range(size) for r in reversed(range(size))),
                        dtype=uint8,
                    )
                    col = array(
                        (c for c in range(size) for _ in range(size)), dtype=uint8
                    )
                case Orientation.DOWN:
                    row = array(
                        (r for r in reversed(range(size)) for _ in range(size)),
                        dtype=uint8,
                    )
                    col = array(
                        (c for _ in range(size) for c in reversed(range(size))),
                        dtype=uint8,
                    )
                case Orientation.RIGHT:
                    row = array(
                        (r for _ in range(size) for r in range(size)), dtype=uint8
                    )
                    col = array(
                        (c for c in reversed(range(size)) for _ in range(size)),
                        dtype=uint8,
                    )
                case _:
                    raise ValueError(f"Invalid orientation: {orientation}")
            indices[orientation.value, 0, :] = row
            indices[orientation.value, 1, :] = col
        return indices

    def orient_local_frame(
        self, local_frame: NDArray[uint8], orientation: Orientation
    ) -> NDArray[uint8]:
        """Rearrange the local frame based on the current orientation of the automaton."""
        orientation_indices = self._orientation_indices[orientation.value]
        return local_frame[orientation_indices[0], orientation_indices[1]]


class AutomatonTypeA(AutomatonBase, radius=1, isize=8):
    """Represents an automaton for the game."""

    eframes: tuple[FrameBase, ...]
    cframes: tuple[FrameBase, ...]
    wrap_mask: int

    @lru_cache(maxsize=None)
    def _eframe_state(
        self, x: int, y: int, orientation: Orientation
    ) -> tuple[bytes, ...]:
        """Return the local environment frame state for the automaton at the given coordinates. This method is cached
        to optimize performance, as the local frame state is likely to be queried multiple times for the
        same coordinates.
        """
        return tuple(
            self.orient_local_frame(e.get(x, y, self.radius), orientation).tobytes()
            for e in self.eframes
        )

    def _cframe_state(
        self, x: int, y: int, orientation: Orientation
    ) -> tuple[bytes, ...]:
        """Return the local creature frame state for the automaton at the given coordinates."""
        # This method is not cached yet because the creature frame state has the potential to change
        # However, if performance becomes an issue, this method can also be cached if we track a `modified` flag.
        return tuple(
            self.orient_local_frame(c.get(x, y, self.radius), orientation).tobytes()
            for c in self.cframes
        )

    def _frame_state(
        self, x: int, y: int, orientation: Orientation
    ) -> tuple[bytes, ...]:
        """Return the local frame state for the automaton at the given coordinates. This method is cached
        to optimize performance, as the local frame state is likely to be queried multiple times for the
        same coordinates.
        """
        return self._eframe_state(x, y, orientation) + self._cframe_state(
            x, y, orientation
        )

    @classmethod
    def set_frames(
        cls, eframes: tuple[FrameBase, ...], cframes: tuple[FrameBase, ...]
    ) -> None:
        """Set the environment and creature frames for the automaton.

        Args:
            eframes: A tuple of Frame instances representing the environment layers.
            cframes: A tuple of Frame instances representing the creature layers.
        """
        # TODO: Yuck - we should have a more robust way to ensure the frames are compatible with
        # the automaton, but for now we will just check that they have the same size and wrap mask.
        assert all(
            e.size == cframes[0].size for e in eframes
        ), "All frames must have the same size"
        cls.eframes = eframes
        cls.cframes = cframes
        cls.wrap_mask = cframes[0].wrap_mask


class AutomatonTypeAIndiviual(AutomatonTypeA):
    """Represents an automaton for the game."""

    def __init__(
        self,
        x: int,
        y: int,
        orientation: Orientation,
        rank: int,
        istate: NDArray[uint8] | None = None,
    ) -> None:
        """Initialize an Automaton instance.

        Args:
            x: The x-coordinate of the automaton's position.
            y: The y-coordinate of the automaton's position.
            orientation: The initial orientation of the automaton.
            rank: The rank of the automaton (used for prioritization or hierarchy).
            istate: The initial internal state of the automaton (optional).
        """
        super().__init__()
        self.x = x
        self.y = y
        self.orientation = orientation
        self.rank = rank
        self.next_action = AutomatonAction.DO_NOTHING

        # `_istates` role is to track the internal state of the automaton across ticks. It
        # acts like a memory, which is used for decision making in the `decide_action`
        # method.
        assert istate is None or istate.shape == (
            self.isize,
        ), "istate must have shape (isize,)"
        self._istate = (
            zeros(self.isize, dtype=uint8) if istate is None else istate.copy()
        )

    def cframe_increment(self) -> None:
        """Increment the value of the current cell in the creature frame."""
        idx = 0
        while self.cframes[idx].grid[self.y, self.x] and idx < len(self.cframes) - 1:
            idx += 1
        if idx < len(self.cframes):
            self.cframes[idx].grid[self.y, self.x] = UINT8_ONE
        # else we are already at the maximum value for the cell, so we do nothing

    def cframe_decrement(self) -> None:
        """Decrement the value of the current cell in the creature frame."""
        idx = len(self.cframes) - 1
        while idx >= 0 and not self.cframes[idx].grid[self.y, self.x]:
            idx -= 1
        if idx >= 0:
            self.cframes[idx].grid[self.y, self.x] = UINT8_ZERO
        # else we are already at the minimum value for the cell, so we do nothing

    def move_forward(self) -> None:
        """Move the automaton forward in the direction it is currently facing."""
        match self.orientation:
            case Orientation.UP:
                self.y = self.y - 1
            case Orientation.RIGHT:
                self.x = self.x + 1
            case Orientation.DOWN:
                self.y = self.y + 1
            case Orientation.LEFT:
                self.x = self.x - 1

    def turn_left(self) -> None:
        """Turn the automaton left (counter-clockwise)."""
        self.orientation = Orientation((self.orientation - 1) & 3)

    def turn_right(self) -> None:
        """Turn the automaton right (clockwise)."""
        self.orientation = Orientation((self.orientation + 1) & 3)

    def teleport(self) -> None:
        """Teleport the automaton to a random location on the grid."""
        # TODO: This is a placeholder implementation. Teleportation should consider the frame state
        # and potentially other things to help find something `interesting`.
        self.x = randrange(self.cframes[0].size)
        self.y = randrange(self.cframes[0].size)

    def zoom_in(self) -> None:
        """Zoom in the automaton's local frame view."""
        pass

    def zoom_out(self) -> None:
        """Zoom out the automaton's local frame view."""
        pass

    def decide_action(self) -> AutomatonAction:
        """Decide the next action for the automaton based on its current state and the local frame state."""
        # This is a placeholder implementation. The actual decision logic should be implemented here.
        self._frame_state(self.x, self.y, self.orientation)
        return AutomatonAction.DO_NOTHING

    def tick(self) -> None:
        """Update the automaton's state for a single tick of the game."""
        match self.next_action:
            case AutomatonAction.DO_NOTHING:
                pass
            case AutomatonAction.CFRAME_INCREMENT:
                self.cframe_increment()
            case AutomatonAction.CFRAME_DECREMENT:
                self.cframe_decrement()
            case AutomatonAction.MOVE_FORWARD:
                self.move_forward()
            case AutomatonAction.TURN_LEFT:
                self.turn_left()
            case AutomatonAction.TURN_RIGHT:
                self.turn_right()
            case AutomatonAction.TELEPORT:
                self.teleport()
            case AutomatonAction.ZOOM_IN:
                self.zoom_in()
            case AutomatonAction.ZOOM_OUT:
                self.zoom_out()
            case _:
                raise NotImplementedError(
                    f"Action {self.next_action} is not implemented yet"
                )

        self.next_action = self.decide_action()
