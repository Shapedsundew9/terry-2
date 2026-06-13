"""Terry module."""
from __future__ import annotations
from numpy import empty, uint8
from numpy.typing import NDArray

class Frame:
    """Represents a single frame of the game state."""

    def __init__(self, size: int = 64, nbits: int = 8, categorical: bool = True) -> None:
        """Initialize a Frame instance.
        
        Args:
            size: The size of the grid (size x size). Default is 64.
            nbits: The number of bits used to represent each cell in the grid. Default is 8,
                allowing for values from 0 to 255.
            categorical: A boolean indicating whether the grid values should be treated as
                categorical (discrete) values. If True, the values in the grid are interpreted as
                distinct categories rather than continuous values. Default is True.
        """
        self.size = size
        self.grid = empty((size, size), dtype=uint8)
        self.nbits = nbits
        self.categorical = categorical
        self.wrap_mask = (1 << nbits) - 1

        # Strict sanity checks to ensure the grid is valid
        assert size.bit_count() == 1, "Size must be a power of 2"
        assert size >= 4, "Size must be >= 4"
        assert size <= 1024, "Size must be <= 1024 to prevent excessive memory usage"
        assert nbits > 0, "Number of bits must be positive"
        assert nbits <= 8, "Number of bits must be 8 or less to fit in uint8"


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
            raise ValueError(f"value {value} is out of valid range for nbits={self.nbits}")
        
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
    

class Automaton():
    """Represents an automaton for the game."""

    def __init__(self, eframes: tuple[Frame, ...], cframes: tuple[Frame, ...], x: int, y: int, radius: int = 1, isize: int = 8) -> None:
        """Initialize an Automaton instance.
        
        Args:
            eframes: A tuple of Frame instances representing the environment layers.
            cframes: A tuple of Frame instances representing the creature layers.
            x: The x-coordinate of the automaton's position.
            y: The y-coordinate of the automaton's position.
            radius: The radius for neighborhood interactions (default is 1).
            isize: The size of the internal state of the automaton in bits (default is 8).
        """
        self.eframes = eframes
        self.cframes = cframes
        self.x = x
        self.y = y
        self.radius = radius
        self.isize = isize
        self._istate = 0

    def tick(self) -> None:
        """Advance the automaton by one tick, updating its internal state based on the environment and creature frames."""
        # Placeholder for the actual logic to update the internal state based on the frames
        # This is where the core behavior of the automaton would be implemented
        pass