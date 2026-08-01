from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, MutableMapping
from math import log
from random import Random
from typing import Any, Callable, Iterator, Self, Sequence

from numpy import array, asarray, dot, int64, random, sum, uint64, where, zeros
from numpy.typing import NDArray

from arc3_agi.checkpoint import SCHEMA_VERSION, Checkpointable

# Constants for Tsetlin Machine implementation
# Precompute powers of 2 for packing boolean results into an integer
_POWERS = asarray([1 << i for i in range(64)], dtype=uint64)
UINT64_ZERO = uint64(0)


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
        code: Any = None,
        seed: int | None = None,
        resp_bits: int = 1,
        missing_key_value_fn: Callable[[], int] | None = None,
    ) -> None:
        """Initialises the genetic code with a given mapping. The mapping can be provided
        as a dictionary or a sequence (implicitly index mapped).
        """
        self._seed = seed
        self._rng = Random(seed)
        self.resp_bits = resp_bits
        self._mkvfn = missing_key_value_fn or (lambda: self._rng.getrandbits(resp_bits))

    def reset_trajectory(self) -> None:
        """Reset any per-episode tracking state.

        Called by AutomatonBase.reset() at the start of each episode/generation.
        The base implementation is a no-op; subclasses that maintain runtime
        state (e.g. a behavioral transition graph) should override this.
        """

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

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return TOML-serialisable metadata for this genetic code."""

    @abstractmethod
    def to_arrays(self) -> dict[str, NDArray[Any]]:
        """Return array payload for NPZ checkpoint storage."""

    @classmethod
    @abstractmethod
    def from_dict(
        cls, d: dict[str, Any], arrays: dict[str, NDArray[Any]], **kwargs: Any
    ) -> GeneticCode:
        """Reconstruct a genetic code instance from checkpoint metadata + arrays."""


class GeneticCodeDict(GeneticCode):
    """A simple implementation of the GeneticCode interface using a dictionary as the underlying
    data structure.
    """

    def __init__(
        self,
        code: Mapping[int, int] | None = None,
        seed: int | None = None,
        resp_bits: int = 1,
    ) -> None:
        super().__init__(code=code, seed=seed, resp_bits=resp_bits)
        self._code: dict[int, int] = dict(code) if code is not None else {}

    def __getitem__(self, key: int) -> int:
        if key not in self._code:
            value = self._mkvfn()
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

    def to_arrays(self) -> dict[str, NDArray[int64]]:
        keys = array(list(self._code.keys()), dtype=int64)
        values = array(list(self._code.values()), dtype=int64)
        return {"keys": keys, "values": values}

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], arrays: dict[str, NDArray[int64]], **kwargs: Any
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
        code: Sequence[int] | None = None,
        seed: int | None = None,
        resp_bits: int = 1,
    ) -> None:
        super().__init__(code=code, seed=seed, resp_bits=resp_bits)
        self._code = list(code) if code is not None else []

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

    def to_arrays(self) -> dict[str, NDArray[int64]]:
        return {"values": array(self._code, dtype=int64)}

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], arrays: dict[str, NDArray[int64]], **kwargs: Any
    ) -> GeneticCodeList:
        values: list[int] = arrays["values"].tolist()
        return cls(values, seed=d.get("seed"), resp_bits=d.get("resp_bits", 1))


class GeneticCodeTsetlin(GeneticCode):
    """A Tsetlin Machine genetic code mapping input bitmasks to response bit vectors

    using multi-clause majority voting.
    """

    def __init__(
        self,
        # Weight matrices for positive and negative literals
        code: (
            tuple[NDArray[uint64], NDArray[uint64]] | None
        ) = None,  # [resp_bits, num_clauses]
        seed: int | None = None,
        resp_bits: int = 1,
        num_clauses: int = 4,
        input_bits: int = 64,
        missing_key_value_fn: Callable[[], int] | None = None,
    ) -> None:
        super().__init__(
            code,
            seed=seed,
            resp_bits=resp_bits,
            missing_key_value_fn=missing_key_value_fn,
        )
        self._np_rng = random.default_rng(self._seed)
        self.num_clauses = num_clauses
        self.input_bits = input_bits
        if self.num_clauses < 1:
            raise ValueError("num_clauses must be at least 1.")
        self.threshold = (num_clauses // 2) + 1

        # ------------------------------------------------------------------
        # Weight Matrix Initialization: Shape = (resp_bits, num_clauses)
        # ------------------------------------------------------------------
        if code is not None:
            self._w_pos = code[0]
            self._w_neg = code[1]
            assert (
                self._w_pos.shape == self._w_neg.shape
            ), "Weight matrices must have the same shape."
            assert (
                self._w_pos.shape[0] == resp_bits
            ), "Weight matrices must match the number of response bits."
            assert (
                self._w_pos.shape[1] == num_clauses
            ), "Weight matrices must match the number of clauses."
            if ((self._w_pos & self._w_neg) != 0).any():
                raise ValueError(
                    "Tsetlin clauses cannot contain contradictory literals."
                )
        else:
            # Initialize sparse random masks if no weights are provided
            # Create empty weight matrices
            self._w_pos = zeros((self.resp_bits, self.num_clauses), dtype=uint64)
            self._w_neg = zeros((self.resp_bits, self.num_clauses), dtype=uint64)

            # Set initial activation probability per bit (e.g., 5% chance a literal is active)
            active_prob = 0.05

            for bit in range(input_bits):
                # Randomly pick state for this bit position across all clauses:
                # 0 = Ignore (00), 1 = Require True (10), 2 = Require False (01)
                # Probability weights: [1 - active_prob, active_prob / 2, active_prob / 2]
                choices = self._np_rng.choice(
                    a=[0, 1, 2],
                    size=(self.resp_bits, self.num_clauses),
                    p=[1.0 - active_prob, active_prob / 2.0, active_prob / 2.0],
                )

                bit_val = uint64(1 << bit)
                self._w_pos |= where(choices == 1, bit_val, UINT64_ZERO)
                self._w_neg |= where(choices == 2, bit_val, UINT64_ZERO)

    def __getitem__(self, key: int) -> int:
        """Evaluates input key L against all clauses across all response bits using

        vectorized NumPy majority voting. Returns packed integer output.
        """
        l_val = uint64(key)

        # 1. Broad-cast clause evaluations across all response bits and clauses
        # Result shapes: (resp_bits, num_clauses) boolean arrays
        match_pos = (self._w_pos & l_val) == self._w_pos
        match_neg = (self._w_neg & l_val) == 0

        # Combine positive and negative literal requirements for every clause
        clauses_satisfied = match_pos & match_neg

        # 2. Count active clauses per response bit along axis 1
        # Result shape: (resp_bits,) integer array containing votes per bit
        votes = sum(clauses_satisfied, axis=1)

        # 3. Apply majority thresholding
        # Result shape: (resp_bits,) boolean array
        bit_results = votes >= self.threshold

        # 4. Pack boolean results directly into a single output integer
        # dot(bit_results, _powers) computes sum(bit[i] * 2^i) in C
        return int(dot(bit_results, _POWERS[: self.resp_bits]))

    def __contains__(self, key: object) -> bool:
        return isinstance(key, int) and 0 <= key < self.input_bits

    def crossover(self, other: GeneticCode, mutation_rate: float = 0.01) -> Self:
        """Combine two unordered clause pools and mutate the resulting child.

        Each positive/negative mask pair is inherited atomically from a random
        clause in either parent. Parents may have different clause counts, but
        must agree on input and response widths. Each child clause independently
        has one literal changed to either of its other two valid states: ignored,
        required true, or required false.

        The first parent's seeded NumPy generator owns all random choices. Its
        The first parent's fixed clause count and strict-majority threshold are
        preserved.
        """
        if not isinstance(other, GeneticCodeTsetlin):
            raise TypeError(
                "GeneticCodeTsetlin can only crossover with another "
                "GeneticCodeTsetlin."
            )
        if self.resp_bits != other.resp_bits:
            raise ValueError("Tsetlin parents must have the same response bits.")
        if self.input_bits != other.input_bits:
            raise ValueError("Tsetlin parents must have the same input bits.")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0 and 1 inclusive.")
        if self.num_clauses < 1 or other.num_clauses < 1:
            raise ValueError("Tsetlin parents must each contain at least one clause.")
        if self.input_bits < 1:
            raise ValueError("Tsetlin parents must have at least one input bit.")

        rng = self._np_rng
        shape = (self.resp_bits, self.num_clauses)
        response_indices = array(range(self.resp_bits), dtype=int64)[:, None]
        self_clause_indices = rng.integers(self.num_clauses, size=shape)
        other_clause_indices = rng.integers(other.num_clauses, size=shape)
        inherit_other = rng.random(shape) < 0.5

        child_w_pos = where(
            inherit_other,
            other._w_pos[response_indices, other_clause_indices],
            self._w_pos[response_indices, self_clause_indices],
        ).astype(uint64, copy=True)
        child_w_neg = where(
            inherit_other,
            other._w_neg[response_indices, other_clause_indices],
            self._w_neg[response_indices, self_clause_indices],
        ).astype(uint64, copy=True)

        if mutation_rate > 0.0:
            mutation_rows, mutation_columns = (
                rng.random(shape) < mutation_rate
            ).nonzero()
            mutation_count = len(mutation_rows)
            if mutation_count:
                mutation_bits = rng.integers(self.input_bits, size=mutation_count)
                alternative_states = rng.integers(2, size=mutation_count)
                bit_masks = uint64(1) << mutation_bits.astype(uint64, copy=False)

                selected_pos = child_w_pos[mutation_rows, mutation_columns].copy()
                selected_neg = child_w_neg[mutation_rows, mutation_columns].copy()
                was_positive = (selected_pos & bit_masks) != 0
                was_negative = (selected_neg & bit_masks) != 0

                keep_mask = ~bit_masks
                selected_pos &= keep_mask
                selected_neg &= keep_mask

                # Incremental transitions for a targeted literal:
                # positive/negative -> ignored, and ignored -> positive/negative.
                choose_alt_one = alternative_states == 1
                was_ignored = ~was_positive & ~was_negative
                set_positive = was_ignored & ~choose_alt_one
                set_negative = was_ignored & choose_alt_one

                selected_pos |= where(set_positive, bit_masks, UINT64_ZERO)
                selected_neg |= where(set_negative, bit_masks, UINT64_ZERO)

                child_w_pos[mutation_rows, mutation_columns] = selected_pos
                child_w_neg[mutation_rows, mutation_columns] = selected_neg

        return self.__class__(
            code=(child_w_pos, child_w_neg),
            seed=int(rng.integers(0, 2**32)),
            resp_bits=self.resp_bits,
            num_clauses=self.num_clauses,
            input_bits=self.input_bits,
        )

    def __setitem__(self, key: int, value: int) -> None:
        raise NotImplementedError(
            "GeneticCodeTsetlin does not support direct item assignment."
        )

    def __delitem__(self, key: int) -> None:
        raise NotImplementedError("GeneticCodeTsetlin does not support item deletion.")

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.input_bits))

    def __len__(self) -> int:
        return self.input_bits

    # ------------------------------------------------------------------
    # Checkpoint interface
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": "GeneticCodeTsetlin",
            "schema_version": SCHEMA_VERSION,
            "resp_bits": self.resp_bits,
            "num_clauses": self.num_clauses,
            "input_bits": self.input_bits,
            "threshold": self.threshold,
        }
        if self._seed is not None:
            d["seed"] = self._seed
        return d

    def to_arrays(self) -> dict[str, NDArray[uint64]]:
        return {"w_pos": self._w_pos, "w_neg": self._w_neg}

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], arrays: dict[str, NDArray[Any]], **kwargs: Any
    ) -> GeneticCodeTsetlin:
        w_pos = arrays["w_pos"].astype(uint64, copy=False)
        w_neg = arrays["w_neg"].astype(uint64, copy=False)
        return cls(
            code=(w_pos, w_neg),
            seed=d.get("seed"),
            resp_bits=d.get("resp_bits", w_pos.shape[0]),
            num_clauses=d.get("num_clauses", w_pos.shape[1]),
            input_bits=d.get("input_bits", 64),
        )
