from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, MutableMapping
from math import log
from random import Random, randint, randrange
from typing import Any, Iterator, Self, Sequence

import numpy as np

from arc3_agi.checkpoint import SCHEMA_VERSION, Checkpointable


class GeneticCode(MutableMapping[int, int], Checkpointable):
    """Represents the genetic code for an automaton species.

    The genetic code simply maps input state to output state and provides some
    utility methods for working with the code and introspection. Optimised
    methods should be implemented for performance.

    Both keys (input codes) and values (output codes) are plain integers whose
    bits encode the packed state/environment/response fields. Using ints avoids
    the per-tick byte allocation and conversion overhead of a bytes-based code.
    """

    @abstractmethod
    def __init__(
        self,
        code: Mapping[int, int] | Sequence[int] | None,
        seed: int | None = None,
        resp_bits: int = 1,
    ) -> None:
        """Initialises the genetic code with a given mapping. The mapping can be provided
        as a dictionary or a sequence (implicitly index mapped).
        """
        self._seed = seed
        self._rng = Random(seed)
        self.resp_bits = resp_bits

    def crossover(self, other: GeneticCode, mutation_rate: float = 0.01) -> Self:
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
            if self._rng.random() < mutation_rate:
                child[key] ^= 1 << self._rng.randrange(self.resp_bits)
        return self.__class__(
            child, seed=self._rng.randint(0, 2**32 - 1), resp_bits=self.resp_bits
        )


class GeneticCodeDict(GeneticCode):
    """A simple implementation of the GeneticCode interface using a dictionary as the underlying
    data structure.
    """

    def __init__(
        self,
        code: Mapping[int, int] | Sequence[int],
        seed: int | None = None,
        resp_bits: int = 1,
    ) -> None:
        super().__init__(code, seed, resp_bits)
        if isinstance(code, Mapping):
            # Copy if the code is a mapping to avoid mutating the original
            self._code: dict[int, int] = dict(code)
        else:
            self._code = {i: r for i, r in enumerate(code)}

    def __getitem__(self, key: int) -> int:
        if key not in self._code:
            value = self._rng.getrandbits(self.resp_bits)
            self._code[key] = value
        return self._code[key]

    def __contains__(self, key: object) -> bool:
        return key in self._code

    def crossover(self, other: GeneticCode, mutation_rate: float = 0.01) -> Self:
        """Combine two parent codes into a child, operating directly on the
        underlying dictionaries.

        For keys present in both parents, the value is inherited from one parent
        at random; keys present in only one parent are inherited from that
        parent. Each inherited entry is then mutated with probability
        ``mutation_rate`` by flipping a single random output bit.
        """
        assert isinstance(
            other, GeneticCodeDict
        ), "GeneticCodeDict can only crossover with another GeneticCodeDict."
        a = self._code
        b = other._code
        rng = self._rng
        rnd = rng.random
        # Start from a copy of this parent, then overlay the other parent.
        child = dict(a)
        for key, vb in b.items():
            if key not in a or rnd() < 0.5:
                child[key] = vb
        resp_bits = self.resp_bits
        if mutation_rate > 0.0:
            randrange = rng.randrange
            keys = list(child)
            n = len(keys)
            # Sample mutation positions from a geometric gap distribution so we
            # draw ~mutation_rate * n randoms instead of one per entry.
            inv_log = 1.0 / log(1.0 - mutation_rate)
            i = int(log(1.0 - rnd()) * inv_log)
            while i < n:
                key = keys[i]
                child[key] ^= 1 << randrange(resp_bits)
                i += 1 + int(log(1.0 - rnd()) * inv_log)
        return self.__class__(
            child, seed=rng.randint(0, 2**32 - 1), resp_bits=resp_bits
        )

    def __setitem__(self, key: int, value: int) -> None:
        self._code[key] = value

    def __delitem__(self, key: int) -> None:
        del self._code[key]

    def __iter__(self) -> Iterator[int]:
        return iter(self._code)

    def __len__(self) -> int:
        return len(self._code)

    # ------------------------------------------------------------------
    # Checkpoint interface
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": "GeneticCodeDict",
            "schema_version": SCHEMA_VERSION,
            "resp_bits": self.resp_bits,
        }
        if self._seed is not None:
            d["seed"] = self._seed
        return d

    def to_arrays(self) -> dict[str, np.ndarray]:
        keys = np.array(list(self._code.keys()), dtype=np.int64)
        values = np.array(list(self._code.values()), dtype=np.int64)
        return {"keys": keys, "values": values}

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], arrays: dict[str, np.ndarray], **kwargs: Any
    ) -> GeneticCodeDict:
        keys: list[int] = arrays["keys"].tolist()
        values: list[int] = arrays["values"].tolist()
        code = dict(zip(keys, values))
        return cls(code, seed=d.get("seed"), resp_bits=d.get("resp_bits", 1))


class GeneticCodeList(GeneticCode):
    """A simple implementation of the GeneticCode interface using a list as the underlying
    data structure. This is more memory efficient for dense codes where the input states are
    contiguous and can be represented as indices.

    NOTE: It is slower than the dictionary due to the overhead of converting keys to indices
    in python. In C++ or rust this would be much faster and more efficient.
    """

    def __init__(
        self,
        code: Mapping[int, int] | Sequence[int],
        seed: int | None = None,
        resp_bits: int = 1,
    ) -> None:
        super().__init__(code, seed, resp_bits)
        if isinstance(code, Mapping):
            self._code = list(code.values())
        else:
            self._code = list(code)

    def __getitem__(self, key: int) -> int:
        return self._code[key]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, int) and 0 <= key < len(self._code)

    def __setitem__(self, key: int, value: int) -> None:
        self._code[key] = value

    def __delitem__(self, key: int) -> None:
        # Need to preserve the indexing, so we can't actually remove items from the list.
        self._code[key] = 0

    def __iter__(self) -> Iterator[int]:
        return iter(range(len(self._code)))

    def __len__(self) -> int:
        return len(self._code)

    # ------------------------------------------------------------------
    # Checkpoint interface
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": "GeneticCodeList",
            "schema_version": SCHEMA_VERSION,
            "resp_bits": self.resp_bits,
        }
        if self._seed is not None:
            d["seed"] = self._seed
        return d

    def to_arrays(self) -> dict[str, np.ndarray]:
        return {"values": np.array(self._code, dtype=np.int64)}

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], arrays: dict[str, np.ndarray], **kwargs: Any
    ) -> GeneticCodeList:
        values: list[int] = arrays["values"].tolist()
        return cls(values, seed=d.get("seed"), resp_bits=d.get("resp_bits", 1))
