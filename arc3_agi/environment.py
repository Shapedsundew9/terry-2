from abc import abstractmethod
from copy import deepcopy
from enum import IntEnum
from typing import Any

from arc3_agi.automaton import ActionStatus, AutomatonBase


class Environment:
    def __init__(self, name: str, **kwargs) -> None:
        self.name = name
        self.description = kwargs.get("description", "No description provided.")

    @abstractmethod
    def get(self, *args, **kwargs) -> Any:
        """Returns the environment in the implementation format."""
        raise NotImplementedError("get method must be implemented by subclasses")

    @abstractmethod
    def get_local(self, coords: list[int], **kwargs) -> bytes:
        """Returns a bitstring in a bytes object representing the local environment at
        the given coordinates.
        """
        raise NotImplementedError("get_local method must be implemented by subclasses")

    @abstractmethod
    def set(self, *args, **kwargs) -> None:
        """Sets the state of the environment. The specific parameters and behavior will depend on the implementation."""
        raise NotImplementedError("set method must be implemented by subclasses")

    @abstractmethod
    def set_local(self, coords: list[int], **kwargs) -> None:
        """Sets the local environment at the given coordinates to the provided value."""
        raise NotImplementedError("set_local method must be implemented by subclasses")


class Boolean2DGrid(Environment):

    class Orientation(IntEnum):
        UP = 0
        RIGHT = 1
        DOWN = 2
        LEFT = 3

    _orientation_offsets_cache: dict[int, list[tuple[list[int], list[int]]]] = {}

    @staticmethod
    def _generate_orientation_offsets(radius: int) -> list[tuple[list[int], list[int]]]:
        r = radius
        offsets = []
        # UP: row by row top-to-bottom (dy -r..r), left-to-right (dx -r..r)
        dy_list, dx_list = [], []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                dy_list.append(dy)
                dx_list.append(dx)
        offsets.append((dy_list, dx_list))
        # RIGHT: column by column right-to-left (dx r..-r), top-to-bottom (dy -r..r)
        dy_list, dx_list = [], []
        for dx in range(r, -r - 1, -1):
            for dy in range(-r, r + 1):
                dy_list.append(dy)
                dx_list.append(dx)
        offsets.append((dy_list, dx_list))
        # DOWN: row by row front-to-back (dy r..-r), left-to-right from automaton's view (dx r..-r)
        dy_list, dx_list = [], []
        for dy in range(r, -r - 1, -1):
            for dx in range(r, -r - 1, -1):
                dy_list.append(dy)
                dx_list.append(dx)
        offsets.append((dy_list, dx_list))
        # LEFT: column by column left-to-right (dx -r..r), bottom-to-top (dy r..-r)
        dy_list, dx_list = [], []
        for dx in range(-r, r + 1):
            for dy in range(r, -r - 1, -1):
                dy_list.append(dy)
                dx_list.append(dx)
        offsets.append((dy_list, dx_list))
        return offsets

    @classmethod
    def get_orientation_offsets(cls, radius: int) -> list[tuple[list[int], list[int]]]:
        if radius not in cls._orientation_offsets_cache:
            cls._orientation_offsets_cache[radius] = cls._generate_orientation_offsets(
                radius
            )
        return cls._orientation_offsets_cache[radius]

    orientation_moves = [
        (0, -1),  # UP
        (1, 0),  # RIGHT
        (0, 1),  # DOWN
        (-1, 0),  # LEFT
    ]

    def __init__(self, name: str, **kwargs) -> None:
        """Initializes a 2D grid environment where each cell can be either True or False.

        Args:
            name: The name of the environment
            description: An optional description of the environment.
        """
        super().__init__(
            name,
            description=kwargs.get(
                "description",
                "A 2D rectangular grid environment where each"
                " cell can be either True or False.",
            ),
        )
        self._map: list[list[bool]] = []

    def get(self) -> list[list[bool]]:
        """Returns the entire 2D grid as a list of lists of booleans.
        NOTE: Modifying the returned *WILL* affect the internal state of the environment.
        """
        return self._map

    def get_local(self, coords: list[int], **kwargs) -> bytes:
        """Returns a bitstring representing the local environment around the
        given coordinates. If wrap is False, out-of-bounds cells will be treated
        as having the value of border_value.

        Args:
            coords: A list of [x, y, orientation] coordinates for the center of
                the local environment. For orientation, 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT.
                The orientation determines the direction in which the local environment
                is considered if positioned in the center. For example, with radius=1,
                the local environment will be a 3x3 grid.
            radius: The radius of the local environment to consider (default: 0, which means
                only the cell at the given coordinates)
            border_value: The value to use for out-of-bounds cells (default: False)
            wrap: Whether to wrap around the edges of the grid (default: False)

        Returns:
            A big endian bytes object representing the local environment as a bitstring, where
            each bit MSb to LSb corresponds to a cell in the local area, ordered from top-left
            to bottom-right.
        """
        radius: int = kwargs.get("radius", 0)
        num_cells = (radius * 2 + 1) ** 2
        border_value: bool = kwargs.get("border_value", False)
        wrap = kwargs.get("wrap", False)
        x, y, orientation = coords
        bitstring: int = 0
        shift = num_cells - 1
        for dy, dx in zip(*self.get_orientation_offsets(radius)[orientation]):
            nx, ny = x + dx, y + dy
            if wrap:
                nx %= len(self._map[0])
                ny %= len(self._map)
            if 0 <= ny < len(self._map) and 0 <= nx < len(self._map[ny]):
                bitstring |= self._map[ny][nx] << shift
            else:
                bitstring |= border_value << shift
            shift -= 1
        return bitstring.to_bytes((num_cells >> 3) + 1, byteorder="big")

    def set(self, new_map: list[list[bool]]) -> None:
        """Sets the entire 2D grid to the provided map. The input should be a list
        of lists of booleans, where each inner list represents a row of the grid."""
        if not all(len(row) == len(new_map[0]) for row in new_map):
            raise ValueError("All rows in the new map must have the same length.")
        self._map = deepcopy(new_map)

    def set_local(self, coords: list[int], **kwargs) -> None:
        """Sets the local environment around the given coordinates to the provided value.
        The specific parameters and behavior will depend on the implementation.

        Args:
            coords: A list of [x, y] coordinates for the center of the local environment
            radius: The radius of the local environment to consider (default: 0, which means
                only the cell at the given coordinates)
            value: The boolean value to set for the local environment (default: True)
            wrap: Whether to wrap around the edges of the grid (default: False)
        """
        radius: int = kwargs.get("radius", 0)
        if radius == 0:
            self._map[coords[1]][coords[0]] = kwargs.get("value", True)
            return
        value: bool = kwargs.get("value", True)
        wrap = kwargs.get("wrap", False)
        x, y = coords
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if wrap:
                    nx %= len(self._map[0])
                    ny %= len(self._map)
                if 0 <= ny < len(self._map) and 0 <= nx < len(self._map[ny]):
                    self._map[ny][nx] = value


class StaticBoolean2DGrid(Boolean2DGrid):

    def __init__(self, name: str, **kwargs) -> None:
        """Initializes a 2D grid environment where each cell can be either True or False
        and the grid cannot be modified after it is initialized.

        Args:
            name: The name of the environment
            description: An optional description of the environment.
            grid: Iterable[Iterable[object]] A 2D arrangement of objects representing the
                state of the grid. This will be set immediately and cannot be modified later.
            radius: The radius to use for get_local and set_local operations (default: 0,
                which means only the cell at the given coordinates)
            border_value: The value to use for out-of-bounds cells in get_local (default: False)
            wrap: Whether to wrap around the edges of the grid in get_local and set_local
                (default: False)
        """
        super().__init__(
            name,
            description=kwargs.get(
                "description",
                "A 2D rectangular grid environment where each"
                " cell can be either True or False. Once set, the grid cannot be modified.",
            ),
        )
        self._map: list[list[bool]] = []
        super().set([[bool(n) for n in row] for row in kwargs.get("grid", [])])
        self._radius: int = kwargs.get("radius", 0)
        self._border_value: bool = kwargs.get("border_value", False)
        self._wrap: bool = kwargs.get("wrap", False)
        self.x_size = len(self._map[0])
        self.y_size = len(self._map)
        self._local_cache: list[list[list[bytes]]] = [
            [
                [
                    super().get_local(
                        [
                            x,
                            y,
                            orientation.value,
                        ],  # orientation doesn't matter for the cache since it's static
                        radius=self._radius,
                        border_value=self._border_value,
                        wrap=self._wrap,
                    )
                    for x in range(self.x_size)
                ]
                for y in range(self.y_size)
            ]
            for orientation in self.Orientation
        ]

    def get_local(self, coords: list[int], **kwargs) -> bytes:
        """Returns a bitstring representing the local environment around the
        given coordinates. This implementation ignores the radius, border_value, and wrap
        parameters since they are fixed at initialization.

        Args:
            coords: A list of [x, y] coordinates for the center of the local environment

        Returns:
            A big endian bytes object representing the local environment as a bitstring, where
            each bit MSb to LSb corresponds to a cell in the local area, ordered from top-left
            to bottom-right.
        """
        return self._local_cache[coords[2]][coords[1]][coords[0]]

    def set(self, new_map: list[list[bool]]) -> None:
        """Sets the entire 2D grid to the provided map. The input should be a list
        of lists of booleans, where each inner list represents a row of the grid.
        This method can only be called once; subsequent calls will raise an error."""
        if self._map:
            raise ValueError(
                "Cannot modify a StaticBoolean2DGrid after it has been set."
            )
        super().set(new_map)

    def set_local(self, coords: list[int], **kwargs) -> None:
        """Sets the local environment around the given coordinates to the provided value.
        This method is not supported for StaticBoolean2DGrid and will raise an error if called.
        """
        raise NotImplementedError(
            "Cannot modify a StaticBoolean2DGrid after it has been set."
        )


class LayeredStaticBoolean2DGrid(Environment):

    Orientation = StaticBoolean2DGrid.Orientation
    orientation_moves = StaticBoolean2DGrid.orientation_moves

    def __init__(self, name: str, **kwargs) -> None:
        """Initializes a layered 2D grid environment where each cell can be either True or False
        and the grid cannot be modified after it is initialized. Each layer represents a different
        aspect of the environment.

        Args:
            name: The name of the environment
            description: An optional description of the environment.
            grid: A Iterable[Iterable[int]] representing the state of each layer as a 2D grid. Each
                layer is a bit position in the integer. This will be set immediately and cannot be
                modified later.
            radius: The radius to use for get_local operations (default: 0, which means only the cell
                at the given coordinates)
            border_value: The value to use for out-of-bounds cells in get_local (default: False)
            wrap: Whether to wrap around the edges of the grid in get_local (default: False)
            num_layers: The number of layers in the grid (the bit width of the integers in the grid).
        """
        self._grid: list[list[int]] = kwargs.get("grid", [])
        num_layers: int = kwargs.get("num_layers", 1)
        if self._grid:
            self.layers: list[StaticBoolean2DGrid] = [
                StaticBoolean2DGrid(
                    f"{name}_layer_{i}",
                    description=f"Layer {i} of the {name} environment.",
                    grid=[[cell & (1 << i) for cell in row] for row in self._grid],
                    radius=kwargs.get("radius", 0),
                    border_value=kwargs.get("border_value", False),
                    wrap=kwargs.get("wrap", False),
                )
                for i in range(num_layers)
            ]

    def add_layer(self, layer: StaticBoolean2DGrid) -> int:
        """Adds a new layer to the environment with the given grid and description.

        Args:
            layer: A StaticBoolean2DGrid representing the new layer to add. The dimensions of this grid
                must match the existing grid.
        Returns:
            The index of the newly added layer.
        """
        if not self._grid:
            self._grid = [[0 for _ in row] for row in layer.get()]
        elif len(layer.get()) != len(self._grid) or len(layer.get()[0]) != len(
            self._grid[0]
        ):
            raise ValueError(
                "New layer must have the same dimensions as the existing grid."
            )
        layer_index = len(self.layers)
        self.layers.append(layer)
        for y in range(len(self._grid)):
            for x in range(len(self._grid[y])):
                if layer.get()[y][x]:
                    self._grid[y][x] |= 1 << layer_index
        return layer_index

    def get(self) -> list[list[int]]:
        """Returns the entire layered grid as a list of lists of integers, where each bit in the integer
        represents a different layer."""
        return self._grid

    def get_local(self, coords: list[int], **kwargs) -> bytes:
        """Returns a bitstring representing the local environment around the
        given coordinates. This implementation ignores the radius, border_value, and wrap
        parameters since they are fixed at initialization.

        Args:
            coords: A list of [x, y] coordinates for the center of the local environment

        Returns:
            A big endian bytes object representing the local environment as a bitstring, where
            each bit MSb to LSb corresponds to a cell in the local area, ordered from top-left
            to bottom-right.
        """
        retval = bytes()
        for layer in self.layers:
            retval += layer.get_local(coords, **kwargs)
        return retval

    def set(self, new_grid: list[list[int]]) -> None:
        """Sets the entire layered grid to the provided grid. The input should be a list
        of lists of integers, where each integer's bits represent the state of each layer for that cell.
        This method can only be called once; subsequent calls will raise an error."""
        raise ValueError(
            "Cannot modify a LayeredStaticBoolean2DGrid after it has been set."
        )

    def set_local(self, coords: list[int], **kwargs) -> None:
        """Sets the local environment around the given coordinates to the provided value.
        This method is not supported for LayeredStaticBoolean2DGrid and will raise an error if called.
        """
        raise NotImplementedError(
            "Cannot modify a LayeredStaticBoolean2DGrid after it has been set."
        )
