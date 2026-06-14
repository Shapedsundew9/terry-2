"""Terry module."""

from __future__ import annotations

from enum import IntEnum
from functools import lru_cache
from random import randrange
from typing import Generic, TypeVar

from numpy import (
    arange,
    dot,
    empty,
    int8,
    int32,
    repeat,
    tile,
    uint8,
    uint16,
    uint32,
    uint64,
    unsignedinteger,
    zeros,
)
from numpy.typing import NDArray

# Constants
UINT8_ZERO = uint8(0)
UINT8_ONE = uint8(1)
UINT16_ZERO = uint16(0)
UINT16_ONE = uint16(1)

# Types
I = TypeVar("I", unsignedinteger, bytes)  # Input type
S = TypeVar("S", unsignedinteger, NDArray[uint64])  # State type
A = TypeVar("A", unsignedinteger, bytes)  # Action type
G = TypeVar("G", unsignedinteger, bytes)  # Genetic map hash type


class GeneticCode(Generic[I, S, A]):
    """Represents the genetic code for the automaton."""

    def __init_subclass__(cls, isize: int, ssize: int, asize: int, **kwargs) -> None:
        """Initialize the subclass with the given parameters."""
        super().__init_subclass__(**kwargs)
        cls.isize = isize
        cls.ssize = ssize
        cls.asize = asize

    def get_state(self, input: I, state: S) -> S:
        """Get the next internal state based on the input and current state."""
        # This method should be implemented by subclasses. The implementation will
        # depend on the specific encoding of the genetic code and how it maps inputs
        # and states to new states. That may be computed or looked up in an indexed
        # table (for I+S state landscapes that can be entirely precomputed), a map
        # (for sparse landscapes), or some other structure.
        raise NotImplementedError("Subclasses must implement the get_state method.")

    def get_action(self, state: S) -> A:
        """Get the action based on the internal state."""
        # This method should be implemented by subclasses. The implementation will
        # depend on the specific encoding of the genetic code and how it maps states
        # to actions. That may be computed or looked up in an indexed table (for
        # state landscapes that can be entirely precomputed), a map (for sparse
        # landscapes), or some other structure.
        raise NotImplementedError("Subclasses must implement the get_action method.")


class AutomatonBase(Generic[I, S, A]):
    """Represents the base class for the automaton.

    An automaton is a computational model that consists of an internal state, a
    set of rules for updating that state based on input and a subsequent action.
    The internal state acts as a memory for the automaton, allowing it to store
    information about the inputs it has received and the actions it has taken.

    Inputs are always a bitstring of length `isize`, the internal state is
    a bitstring of length `ssize` and an action represented by an integer
    0 <= action < `asize`.

    Each 'tick' of the automaton consists of the following steps:
        1. The automaton receives an input.
        2. The automaton updates its internal state based on the input and its
            current state.
        3. The automaton produces an action based on its updated internal state.

    The behavior of the automaton is determined by its genetic code, which
    encodes the rules for updating the internal state and producing actions.

    An Automaton only ever updates its internal state directly. The impact
    of the action is determined by the environment in which the automaton is.
    """

    def __init_subclass__(cls, isize: int, ssize: int, asize: int, **kwargs) -> None:
        """Initialize the subclass with the given parameters.
        Args:
            isize: The size of the automaton's input in bits
            ssize: The size of the automaton's internal state in bits
            asize: The size of the automaton's action space in bits
        """
        super().__init_subclass__(**kwargs)
        cls.isize = isize
        cls.ssize = ssize
        cls.asize = asize

    def __init__(self, genetic_code: GeneticCode[I, S, A], state: S) -> None:
        """Initialize the automaton."""
        self._state = state
        self.genetic_code = genetic_code

    def _tick(self, input: I) -> A:
        """Perform a tick of the automaton."""
        self._state = self.genetic_code.get_state(input, self._state)
        return self.genetic_code.get_action(self._state)

    def tick(self) -> None:
        """Perform a tick of the automaton."""
        raise NotImplementedError("Subclasses must implement the tick method.")


class Automaton2DGrid(
    AutomatonBase[uint32, uint16, uint8],
    isize=9 * 2,
    ssize=9,
    asize=3,
):
    class AutomatonAction(IntEnum):
        """Represents the possible actions for the automaton."""

        MOVE_FORWARD = 0
        TURN_LEFT = 1
        TURN_RIGHT = 2

    class Orientation(IntEnum):
        """Represents the possible orientations for the automaton."""

        UP = 0
        RIGHT = 1
        DOWN = 2
        LEFT = 3

    def __init_subclass__(
        cls, radius: int, environment: Environment2DGrid, **kwargs
    ) -> None:
        """Initialize the subclass with the given parameters."""
        super().__init_subclass__(**kwargs)
        cls.radius = radius
        cls.environment = environment
        cls.orientation_indices = cls._generate_orientation_indices()

        # Used for efficient bitshifting when converting the local environment to a uint32
        # input for the automaton.
        cls.h9powers = 2 ** arange(9, 18, dtype=uint32)
        cls.l9powers = 2 ** arange(9, dtype=uint32)

    @classmethod
    def _generate_orientation_indices(cls) -> NDArray[int32]:
        """An Automaton only sees the local frame in its current orientation. This method generates
        a mapping of relative grid coordinates to indices for all orientations as a reference.

        Using the example of a 3x3 grid (radius 1 from the centre x, y) the relative
        grid coordinates are:
            (x-1, y-1) (x, y-1) (x+1, y-1)
            (x-1, y)   (x, y)   (x+1, y)
            (x-1, y+1) (x, y+1) (x+1, y+1)
        Converting to numpy [row, column] indexing
            row [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            col [-1, 0, 1, -1, 0, 1, -1, 0, 1]
        The numpy [row, column] indexing for "LEFT" orientation is effectively a 90 degree clockwise
        rotation of the "UP" orientation, so the indexing becomes:
            row [1, 0, -1, 1, 0, -1, 1, 0, -1]
            col [-1, -1, -1, 0, 0, 0, 1, 1, 1]
        "DOWN" orientation is a 180 degree rotation of "UP":
            row [1, 1, 1, 0, 0, 0, -1, -1, -1]
            col [1, 0, -1, 1, 0, -1, 1, 0, -1]
        "RIGHT" orientation is a 270 degree rotation of "UP":
            row [-1, 0, 1, -1, 0, 1, -1, 0, 1]
            col [1, 1, 1, 0, 0, 0, -1, -1, -1]

        The returned array has shape (4, 2, (radius*2 + 1)**2) where the first dimension corresponds to the
        orientation and the second dimension corresponds row, column and the third dimension
        corresponds to the flattened indices.
        """
        size = cls.radius * 2 + 1
        r = arange(-cls.radius, cls.radius + 1)
        row = repeat(r, size)  # slow axis: [-R..-R, ..., R..R]
        col = tile(r, size)  # fast axis: [-R..R, -R..R, ...]
        # Successive 90° CW rotations: (r, c) → (c, −r) → (−r, −c) → (−c, r)
        # Matches Orientation enum: UP=0, RIGHT=1, DOWN=2, LEFT=3
        rotations = [(row, col), (col, -row), (-row, -col), (-col, row)]
        indices = empty((4, 2, size * size), dtype=int32)
        for i, (r_i, c_i) in enumerate(rotations):
            indices[i, 0] = r_i
            indices[i, 1] = c_i
        return indices


class EnvironmentBase:
    """Represents the environment in which automata operate.

    The environment can be considered the 'world' or 'physics simulation engine'
    in which the automaton exists. It defines how the automaton's actions affect
    the state of the world. The environment is responsible for updating the
    automaton's relevant input variables.

    Environments consist of layers of binary properties that can be observed and/or
    acted upon by the automaton. For example, a simple 2D block maze world might
    consist of a layer representing the presence or absence of walls and a layer
    representing the presence or absence of the goal. These layers could be
    considered immutable properties of the world that the automaton can observe
    but not change but the environment may also have mutable layers for example
    representing bread crumbs dropped (or collected) by an automaton etc.

    FUTURE: Sophisticated evolutionary environments could allow layers to be added
    (and removed) as automata's evolve.
    """

    def __init__(self) -> None:
        """Initialize the environment."""
        self.ilayers: dict[IntEnum, NDArray[uint8]] = (
            {}
        )  # Dictionary of immutable layers
        self.mlayers: dict[IntEnum, NDArray[uint8]] = {}  # Dictionary of mutable layers


class Environment2D(EnvironmentBase):
    """Represents a 2D environment for the automaton."""

    def __init__(self, width: int, height: int, wrap: bool) -> None:
        """Initialize the 2D environment.

        Args:
            width: The width of the environment
            height: The height of the environment
            wrap: Whether the environment wraps around at the edges
              (i.e., is a torus)
        """
        super().__init__()
        self.width = width
        self.height = height
        self.wrap = wrap


class Environment2DGrid(Environment2D):
    """Represents a square 2D grid environment that wraps around at
    the edges (i.e., a torus)."""

    class LKEYS(IntEnum):
        WALL = 0
        GOAL = 1

    def __init__(self, side_length_bits: int) -> None:
        """Initialize the 2D grid torus environment.

        Args:
            side_length_bits: The side length of the square grid in bits.
                i.e. the side of the grid is 2^side_length_bits.
        """
        super().__init__(
            width=2**side_length_bits, height=2**side_length_bits, wrap=True
        )
        self.wrap_mask = 2**side_length_bits - 1  # Mask for wrapping coordinates

    def add_layer(self, key: IntEnum, mutable: bool = False) -> None:
        """Add a layer to the environment.

        Args:
            key: The key for the layer.
            mutable: Whether the layer is mutable (default: False)
        """
        # Each layer is represented as a 2D array of uint8 values each representing
        # a cell property (bit) in the grid.
        layer = zeros((self.height, self.width), dtype=uint8)
        if mutable:
            self.mlayers[key] = layer
        else:
            self.ilayers[key] = layer


class GeneticCode2DGrid(
    GeneticCode[uint32, uint16, uint8], isize=9 * 2, ssize=9, asize=3
):
    """Represents the genetic code for the automaton in a 2D grid environment."""

    def __init__(self, isize: int, ssize: int, asize: int) -> None:
        """Initialize the genetic code."""
        super().__init__()
        # The genetic code is represented as a 2D array of uint8 values where the
        # first dimension corresponds to the input and the second dimension
        # corresponds to the state. The value at each position is the action to
        # take for that input and state.
        self.code = zeros((2**isize, 2**ssize), dtype=uint8)


class Automaton(
    Automaton2DGrid, radius=1, environment=Environment2DGrid(side_length_bits=6)
):
    """Represents the automaton for Terry's world."""

    def __init__(
        self,
        genetic_code: GeneticCode2DGrid,
        state: uint16,
        x: int32,
        y: int32,
        orientation: Automaton.Orientation,
    ) -> None:
        """Initialize the automaton."""
        super().__init__(genetic_code, state)
        self.x = x
        self.y = y
        self.orientation = orientation
        self.wall_layer = self.environment.ilayers[self.environment.LKEYS.WALL]
        self.goal_layer = self.environment.ilayers[self.environment.LKEYS.GOAL]

    def move_forward(self) -> None:
        """Move the automaton forward in the direction it is currently facing."""
        match self.orientation:
            case self.Orientation.UP:
                self.y = (self.y - 1) & self.environment.wrap_mask
            case self.Orientation.RIGHT:
                self.x = (self.x + 1) & self.environment.wrap_mask
            case self.Orientation.DOWN:
                self.y = (self.y + 1) & self.environment.wrap_mask
            case self.Orientation.LEFT:
                self.x = (self.x - 1) & self.environment.wrap_mask

    def take_action(self, action: uint8) -> None:
        """Perform the given action."""
        match self.AutomatonAction(action):
            case self.AutomatonAction.MOVE_FORWARD:
                self.move_forward()
            case self.AutomatonAction.TURN_LEFT:
                self.turn_left()
            case self.AutomatonAction.TURN_RIGHT:
                self.turn_right()
            case _:
                raise ValueError(f"Invalid action: {action}")

    def tick(self) -> None:
        """Perform a tick of the automaton."""
        # Get the local environment correctly oriented for each layer
        # TODO: Maybe instead of all this indexing and bitshifting on every tick we
        # make orientation part of the automaton  could maintain a "local view" of the environment
        orientation_indices = self.orientation_indices[self.orientation.value]
        wrap_mask = self.environment.wrap_mask
        orientation_indices[0] = (orientation_indices[0] + self.y) & wrap_mask
        orientation_indices[1] = (orientation_indices[1] + self.x) & wrap_mask
        local_walls = self.wall_layer[orientation_indices[0], orientation_indices[1]]
        local_goals = self.goal_layer[orientation_indices[0], orientation_indices[1]]
        input = dot(self.h9powers, local_walls) + dot(self.l9powers, local_goals)

        # Perform a tick of the automaton with the input and get the action
        self.take_action(self._tick(input))

    def turn_left(self) -> None:
        """Turn the automaton left (counter-clockwise)."""
        self.orientation = self.Orientation((self.orientation.value - 1) & 3)

    def turn_right(self) -> None:
        """Turn the automaton right (clockwise)."""
        self.orientation = self.Orientation((self.orientation.value + 1) & 3)
