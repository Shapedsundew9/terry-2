from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, MutableMapping
from random import Random, randrange
from typing import Iterator, Self, Sequence


class GeneticCode(MutableMapping[bytes, bytes]):
    """Represents the genetic code for an automaton species.

    The genetic code simply maps input state to output state and provides some
    utility methods for working with the code and introspection. Optimised
    methods should be implemented for performance.
    """

    @abstractmethod
    def __init__(
        self, code: Mapping[bytes, bytes] | Sequence[bytes], seed: int | None = None
    ) -> None:
        """Initialises the genetic code with a given mapping. The mapping can be provided
        as a dictionary or a sequence (implicitly index mapped).
        """
        self._seed = seed
        self._rng = Random(seed)

    def crossover(self, other: GeneticCode) -> Self:
        """Performs a crossover between this genetic code and another, producing a new genetic
        code that combines elements of both parents. The crossover point is randomly
        selected, and the resulting code is a combination of the two parent codes.
        """
        child = {}
        # Crossover the smap by randomly choosing entries from either parent
        for key in set(self.keys()).union(other.keys()):
            if key not in self:
                child[key] = other[key]
            elif key not in other:
                child[key] = self[key]
            else:
                child[key] = self[key] if self._rng.randrange(2) == 0 else other[key]
        return self.__class__(child, seed=self._rng.randint(0, 2**32 - 1))


class GeneticCodeDict(GeneticCode):
    """A simple implementation of the GeneticCode interface using a dictionary as the underlying
    data structure.
    """

    def __init__(
        self, code: Mapping[bytes, bytes] | Sequence[bytes], seed: int | None = None
    ) -> None:
        super().__init__(code, seed)
        if isinstance(code, Mapping):
            self._code: dict[bytes, bytes] = dict(code)
        else:
            num_bytes = (
                len(code) >> 8
            ) + 1  # Calculate size needed to represent the indices
            self._code = {i.to_bytes(num_bytes, "big"): r for i, r in enumerate(code)}

    def __getitem__(self, key: bytes) -> bytes:
        return self._code[key]

    def __setitem__(self, key: bytes, value: bytes) -> None:
        self._code[key] = value

    def __delitem__(self, key: bytes) -> None:
        del self._code[key]

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._code)

    def __len__(self) -> int:
        return len(self._code)


class GeneticCodeList(GeneticCode):
    """A simple implementation of the GeneticCode interface using a list as the underlying
    data structure. This is more memory efficient for dense codes where the input states are
    contiguous and can be represented as indices.

    NOTE: It is slower than the dictionary due to the overhead of converting keys to indices
    in python. In C++ or rust this would be much faster and more efficient.
    """

    def __init__(
        self, code: Mapping[bytes, bytes] | Sequence[bytes], seed: int | None = None
    ) -> None:
        super().__init__(code, seed)
        if isinstance(code, Mapping):
            self._code = list(code.values())
        else:
            self._code = list(code)
        self._index_size = (len(self._code) >> 8) + 1

    def __getitem__(self, key: bytes) -> bytes:
        return self._code[int.from_bytes(key, "big")]

    def __setitem__(self, key: bytes, value: bytes) -> None:
        self._code[int.from_bytes(key, "big")] = value

    def __delitem__(self, key: bytes) -> None:
        # Need to preserve the indexing, so we can't actually remove items from the list.
        self._code[int.from_bytes(key, "big")] = b"\x00" * len(key)

    def __iter__(self) -> Iterator[bytes]:
        return (i.to_bytes(self._index_size, "big") for i in range(len(self._code)))

    def __len__(self) -> int:
        return len(self._code)
