from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping, MutableMapping
from math import log
from random import Random, randint, randrange
from typing import Iterator, Self, Sequence

# A single node in a ``GeneticCodeGraph``: (op, in_a, in_b, const).
#   op    - one of the _OP_* operation codes (defined below).
#   in_a  - value-array index of the first operand (0 == raw input key).
#   in_b  - value-array index of the second operand (binary ops only).
#   const - evolvable integer operand used by shifts/mask/const ops and as the
#           shift amount source.
Node = tuple[int, int, int, int]
# The full graph genome passed as ``code`` is simply the sequence of nodes; the
# output is read from the last few nodes (see GeneticCodeGraph).
Genome = Sequence[Node]


class GeneticCode(MutableMapping[int, int]):
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
        code: Mapping[int, int] | Sequence[int] | Genome | None,
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


# Logical operations available to a ``GeneticCodeGraph`` node. Kept as plain
# module-level integers (rather than an IntEnum) so the evaluation hot loop can
# compare ints directly without the IntEnum lookup overhead.
_OP_AND = 0  # a & b
_OP_OR = 1  # a | b
_OP_XOR = 2  # a ^ b
_OP_NAND = 3  # ~(a & b)
_OP_NOR = 4  # ~(a | b)
_OP_XNOR = 5  # ~(a ^ b)
_OP_NOT = 6  # ~a (unary)
_OP_IDENTITY = 7  # a (unary)
_OP_SHL = 8  # a << (const % working_bits)
_OP_SHR = 9  # a >> (const % working_bits)
_OP_MASK = 10  # a & const
_OP_CONST = 11  # const
_OP_ROL = 12  # rotate a left by (const % working_bits) within working_bits
_OP_ROR = 13  # rotate a right by (const % working_bits) within working_bits
_NUM_OPS = 14


class GeneticCodeGraph(GeneticCode):
    """A :class:`GeneticCode` whose input→output mapping is computed by a graph
    of logical bit operations rather than stored in a lookup table.

    Instead of memorising an output for every input, the response for a key is
    *computed* by feeding the key through a directed acyclic graph (DAG) of bit
    operations (shifts, masks, AND/OR/XOR/NOT, ...). Because the mapping is a
    deterministic function of the input bits, structurally related keys tend to
    produce related outputs, and the genome size is independent of the input
    space size (so it scales to large input widths).

    Genome layout
    -------------
    The graph is a fixed-length list of :data:`Node` tuples. Evaluation uses a
    value array where index ``0`` holds the (masked) input key and node ``p``
    writes its result to value-array index ``p + 1``. A node may only reference
    value-array indices ``0 .. p`` (the input or strictly earlier nodes), which
    guarantees the graph is acyclic and that index-aligned crossover between two
    same-sized genomes is always valid. The output is the XOR of the last
    ``output_fanin`` node values (masked to ``resp_bits``); folding several of
    the deepest, most-mixed nodes keeps the initial random mapping strongly
    dependent on the input instead of frequently collapsing to a constant.

    All intermediate values are masked to ``working_bits`` (``max(input_bits,
    resp_bits)``) so shifts cannot grow integers without bound. Constants are
    bounded to the same width, and shift/rotate amounts are taken modulo
    ``working_bits``. Rotations (``ROL``/``ROR``) preserve every bit so, unlike
    plain shifts, they never annihilate the input.

    Note
    ----
    The class is a function, not a table, so the ``MutableMapping`` mutators
    (``__setitem__`` / ``__delitem__`` / ``__iter__`` / ``__len__``) are not
    meaningful and raise :class:`NotImplementedError`. Only ``__getitem__`` and
    :meth:`crossover` are used at runtime.
    """

    def __init__(
        self,
        code: Genome | None = None,
        seed: int | None = None,
        resp_bits: int = 1,
        input_bits: int = 1,
        num_nodes: int | None = None,
    ) -> None:
        """Initialise the graph genetic code.

        Args:
            code: The genome as a sequence of :data:`Node` tuples. If ``None`` a
                random graph of ``num_nodes`` nodes is generated using ``seed``.
            seed: Optional seed for the internal RNG (random generation,
                crossover and mutation).
            resp_bits: Total output width in bits (the returned value is masked
                to this width). For an :class:`AutomatonISBase` this is
                ``state_bits + resp_bits``.
            input_bits: Width in bits of the input key. The key is masked to this
                width before evaluation.
            num_nodes: Number of nodes when generating a random graph. Defaults
                to ``max(8, (input_bits + resp_bits) * 2)``. Ignored when ``code``
                is provided.
        """
        super().__init__(code, seed, resp_bits)
        self.input_bits = input_bits
        self.input_mask = (1 << input_bits) - 1
        self.working_bits = max(input_bits, resp_bits, 1)
        self.working_mask = (1 << self.working_bits) - 1
        self.resp_mask = (1 << resp_bits) - 1
        if code is None:
            if num_nodes is None:
                num_nodes = max(8, (input_bits + resp_bits) * 2)
            self._nodes: list[Node] = self._random_nodes(num_nodes)
        else:
            self._nodes = [(int(op), int(a), int(b), int(c)) for op, a, b, c in code]
        self.num_nodes = len(self._nodes)
        # Number of trailing nodes XOR-folded to form the output. Tying it to the
        # output width keeps roughly one output source per output bit.
        self.output_fanin = min(self.num_nodes, max(4, resp_bits))
        self._output_start = self.num_nodes + 1 - self.output_fanin
        # Per-instance memoisation. The mapping is a pure function, so caching
        # results for repeated keys (common across automaton ticks) is safe and
        # bounded by 2 ** input_bits distinct keys.
        self._cache: dict[int, int] = {}

    @classmethod
    def random(
        cls,
        input_bits: int,
        resp_bits: int,
        num_nodes: int | None = None,
        seed: int | None = None,
    ) -> Self:
        """Construct a random graph genetic code. Thin wrapper over ``__init__``
        with ``code=None`` that places ``input_bits`` and ``resp_bits`` first for
        readability at call sites."""
        return cls(
            None,
            seed=seed,
            resp_bits=resp_bits,
            input_bits=input_bits,
            num_nodes=num_nodes,
        )

    def _random_nodes(self, num_nodes: int) -> list[Node]:
        """Generate ``num_nodes`` random nodes obeying the acyclic back-reference
        constraint (node ``p`` may reference value indices ``0 .. p``)."""
        rng = self._rng
        working_bits = self.working_bits
        nodes: list[Node] = []
        for p in range(num_nodes):
            op = rng.randrange(_NUM_OPS)
            in_a = rng.randrange(p + 1)
            in_b = rng.randrange(p + 1)
            const = rng.getrandbits(working_bits)
            nodes.append((op, in_a, in_b, const))
        return nodes

    def _evaluate(self, key: int) -> int:
        """Evaluate the graph for ``key`` and return the masked output."""
        nodes = self._nodes
        wm = self.working_mask
        wb = self.working_bits
        vals = [0] * (len(nodes) + 1)
        vals[0] = key & self.input_mask
        for p, (op, a, b, const) in enumerate(nodes):
            va = vals[a]
            if op == _OP_AND:
                r = va & vals[b]
            elif op == _OP_OR:
                r = va | vals[b]
            elif op == _OP_XOR:
                r = va ^ vals[b]
            elif op == _OP_NAND:
                r = ~(va & vals[b])
            elif op == _OP_NOR:
                r = ~(va | vals[b])
            elif op == _OP_XNOR:
                r = ~(va ^ vals[b])
            elif op == _OP_NOT:
                r = ~va
            elif op == _OP_IDENTITY:
                r = va
            elif op == _OP_SHL:
                r = va << (const % wb)
            elif op == _OP_SHR:
                r = va >> (const % wb)
            elif op == _OP_MASK:
                r = va & const
            elif op == _OP_CONST:
                r = const
            elif op == _OP_ROL:
                n = const % wb
                r = va if n == 0 else (va << n) | (va >> (wb - n))
            else:  # _OP_ROR
                n = const % wb
                r = va if n == 0 else (va >> n) | (va << (wb - n))
            vals[p + 1] = r & wm
        out = 0
        for i in range(self._output_start, len(vals)):
            out ^= vals[i]
        return out & self.resp_mask

    def __getitem__(self, key: int) -> int:
        cache = self._cache
        if key in cache:
            return cache[key]
        value = self._evaluate(key)
        cache[key] = value
        return value

    def crossover(self, other: GeneticCode, mutation_rate: float = 0.01) -> Self:
        """Combine two parent graphs into a child via index-aligned crossover.

        Both parents must be :class:`GeneticCodeGraph` instances with matching
        ``num_nodes`` and ``input_bits`` so that node back-references stay valid.
        For each node index the child inherits the node from one parent at
        random. Each inherited gene (operation, either connection and the
        constant) is then mutated independently with probability
        ``mutation_rate``.
        """
        assert isinstance(
            other, GeneticCodeGraph
        ), "GeneticCodeGraph can only crossover with another GeneticCodeGraph."
        assert (
            self.num_nodes == other.num_nodes
        ), "GeneticCodeGraph crossover requires both parents to have the same number of nodes."
        assert (
            self.input_bits == other.input_bits
        ), "GeneticCodeGraph crossover requires both parents to have the same input width."
        rng = self._rng
        rnd = rng.random
        a = self._nodes
        b = other._nodes
        working_bits = self.working_bits
        mutate = mutation_rate > 0.0
        child_nodes: list[Node] = []
        for p in range(self.num_nodes):
            op, in_a, in_b, const = a[p] if rnd() < 0.5 else b[p]
            if mutate:
                if rnd() < mutation_rate:
                    op = rng.randrange(_NUM_OPS)
                if rnd() < mutation_rate:
                    in_a = rng.randrange(p + 1)
                if rnd() < mutation_rate:
                    in_b = rng.randrange(p + 1)
                if rnd() < mutation_rate:
                    const ^= 1 << rng.randrange(working_bits)
            child_nodes.append((op, in_a, in_b, const))
        return self.__class__(
            tuple(child_nodes),
            seed=rng.randint(0, 2**32 - 1),
            resp_bits=self.resp_bits,
            input_bits=self.input_bits,
        )

    # -- Mapping mutators are not meaningful for a function-based code. --
    def __setitem__(self, key: int, value: int) -> None:
        raise NotImplementedError(
            "GeneticCodeGraph is a function, not a table; entries cannot be set."
        )

    def __delitem__(self, key: int) -> None:
        raise NotImplementedError(
            "GeneticCodeGraph is a function, not a table; entries cannot be deleted."
        )

    def __iter__(self) -> Iterator[int]:
        raise NotImplementedError(
            "GeneticCodeGraph maps an unbounded key space and is not iterable."
        )

    def __len__(self) -> int:
        raise NotImplementedError(
            "GeneticCodeGraph maps an unbounded key space and has no length."
        )
