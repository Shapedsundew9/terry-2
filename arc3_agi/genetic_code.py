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


class GeneticCodeSCC(GeneticCodeDict):
    """A GeneticCodeDict that performs SCC-aware crossover.

    The standard per-entry crossover ignores the epistatic dependencies between
    state machine entries — the new-state output of one entry determines which
    entry fires on the next tick, so mixing entries from two parents can create
    pathological loops that neither parent exhibited.

    This subclass builds the state-transition graph for each parent (edges from
    input-state → output-state, env bits ignored), partitions it into Strongly
    Connected Components (SCCs) using Tarjan's algorithm, then transplants a
    random subset of SCCs from parent B into parent A.  Transplanted states are
    remapped to fresh IDs that do not overlap with parent A's state space, and
    cross-SCC dangling edges are re-wired to a random entry-point state within
    the target SCC.  Standard per-entry point-mutation is applied afterwards.

    Args:
        code:       Initial mapping (as for GeneticCodeDict).
        env_bits:   Number of bits used to encode the environment stimulus in
                    each key (key >> env_bits == input_state).
        state_bits: Number of bits used to encode the internal state in each
                    output value (value & state_mask == output_state).
        seed:       Optional RNG seed.
        resp_bits:  Total output-bit width (state_bits + action_bits).
    """

    def __init__(
        self,
        code: Mapping[int, int] | Sequence[int],
        env_bits: int = 0,
        state_bits: int = 0,
        seed: int | None = None,
        resp_bits: int = 1,
    ) -> None:
        super().__init__(code, seed=seed, resp_bits=resp_bits)
        self.env_bits = env_bits
        self.state_bits = state_bits
        self._state_mask = (1 << state_bits) - 1 if state_bits > 0 else 0

    # ------------------------------------------------------------------
    # SCC helpers
    # ------------------------------------------------------------------

    def _build_state_graph(self) -> dict[int, set[int]]:
        """Return adjacency map {input_state: {output_state, ...}} from the code."""
        graph: dict[int, set[int]] = {}
        for key, value in self._code.items():
            src = key >> self.env_bits
            dst = value & self._state_mask
            graph.setdefault(src, set()).add(dst)
            graph.setdefault(dst, set())  # ensure every node is present
        return graph

    @staticmethod
    def _tarjan_sccs(graph: dict[int, set[int]]) -> list[list[int]]:
        """Tarjan's SCC algorithm.  Returns a list of SCCs (each a list of node IDs)."""
        index_counter = [0]
        stack: list[int] = []
        lowlink: dict[int, int] = {}
        index: dict[int, int] = {}
        on_stack: dict[int, bool] = {}
        sccs: list[list[int]] = []

        def strongconnect(v: int) -> None:
            index[v] = lowlink[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack[v] = True

            for w in graph.get(v, set()):
                if w not in index:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif on_stack.get(w, False):
                    lowlink[v] = min(lowlink[v], index[w])

            if lowlink[v] == index[v]:
                scc: list[int] = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)

        for v in graph:
            if v not in index:
                strongconnect(v)

        return sccs

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def crossover(self, other: GeneticCode, mutation_rate: float = 0.01) -> GeneticCodeSCC:
        """SCC-aware crossover between *self* (parent A) and *other* (parent B).

        Algorithm
        ---------
        1. Build state-transition graphs for both parents and find their SCCs.
        2. Randomly pick a subset of SCCs from parent B to transplant.
        3. Remap transplanted state IDs to fresh IDs beyond parent A's max state.
        4. Stitch dangling cross-SCC edges to a random entry-point in the target SCC.
        5. Apply per-entry point-mutation (geometric sampling, same as GeneticCodeDict).
        """
        assert isinstance(other, GeneticCodeSCC), (
            "GeneticCodeSCC can only crossover with another GeneticCodeSCC."
        )
        assert other.env_bits == self.env_bits and other.state_bits == self.state_bits, (
            "Both parents must have the same env_bits and state_bits."
        )

        rng = self._rng

        # ---- Step 1: SCCs for both parents --------------------------------
        graph_a = self._build_state_graph()
        graph_b = other._build_state_graph()
        sccs_b = self._tarjan_sccs(graph_b)

        # Decide which SCCs from B to transplant (each independently with p=0.5).
        transplant_sccs = [scc for scc in sccs_b if rng.random() < 0.5]

        # ---- Step 2: Build remap table for transplanted B states ----------
        # Fresh IDs start above the maximum state ID currently in parent A's code.
        max_state_a = max(
            (key >> self.env_bits for key in self._code),
            default=-1,
        )
        max_state_a = max(max_state_a, self._state_mask)  # at least the full mask width

        next_fresh = max_state_a + 1
        b_state_remap: dict[int, int] = {}
        for scc in transplant_sccs:
            for state in scc:
                b_state_remap[state] = next_fresh
                next_fresh += 1

        transplanted_b_states: set[int] = set(b_state_remap)

        # ---- Step 3: Build SCC → member set for both parents (for stitching) --
        # For stitching we need: given a state in B, which SCC does it belong to?
        b_state_to_scc: dict[int, list[int]] = {}
        for scc in sccs_b:
            for state in scc:
                b_state_to_scc[state] = scc

        # Build a lookup: remapped-state → members-of-same-remapped-SCC
        remapped_scc_members: dict[int, list[int]] = {}
        for scc in transplant_sccs:
            remapped = [b_state_remap[s] for s in scc]
            for r in remapped:
                remapped_scc_members[r] = remapped

        # Collect entries from A whose input-state is NOT being replaced,
        # plus all entries from A by default.
        child_code: dict[int, int] = dict(self._code)

        # ---- Step 4: Inject transplanted B entries (with remapped state IDs) ---
        env_bits = self.env_bits
        state_mask = self._state_mask
        state_bits = self.state_bits
        env_mask = (1 << env_bits) - 1

        for key_b, val_b in other._code.items():
            src_state_b = key_b >> env_bits
            if src_state_b not in transplanted_b_states:
                continue
            env_stimulus = key_b & env_mask
            new_src = b_state_remap[src_state_b]
            new_key = (new_src << env_bits) | env_stimulus

            dst_state_b = val_b & state_mask
            action_bits_val = val_b >> state_bits

            if dst_state_b in b_state_remap:
                # Destination is also transplanted — remap directly.
                new_dst = b_state_remap[dst_state_b]
            else:
                # Dangling edge: destination is a state NOT being transplanted.
                # Re-wire to a random state in parent A's graph (any state in A).
                a_states = list(graph_a.keys()) or [0]
                new_dst = rng.choice(a_states) & state_mask
            new_val = (action_bits_val << state_bits) | new_dst
            child_code[new_key] = new_val

        # ---- Step 5: Fix dangling edges in A that pointed to replaced states ---
        # A's entries that point to states which now have been transplanted-from B
        # must be re-wired.  (We did not remove those states from A, so this only
        # matters if A had edges into states that conceptually belong to a swapped
        # SCC — but since we ADD new SCCs rather than removing A's states, A's
        # existing entries remain valid.  No repair needed for A→A edges.)

        # ---- Step 6: Apply per-entry point-mutation -----------------------
        if mutation_rate > 0.0:
            resp_bits = self.resp_bits
            rnd = rng.random
            inv_log = 1.0 / log(1.0 - mutation_rate)
            keys = list(child_code)
            n = len(keys)
            i = int(log(1.0 - rnd()) * inv_log)
            while i < n:
                k = keys[i]
                child_code[k] ^= 1 << rng.randrange(resp_bits)
                i += 1 + int(log(1.0 - rnd()) * inv_log)

        return GeneticCodeSCC(
            child_code,
            env_bits=env_bits,
            state_bits=state_bits,
            seed=rng.randint(0, 2**32 - 1),
            resp_bits=self.resp_bits,
        )

    # ------------------------------------------------------------------
    # Checkpoint interface
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["type"] = "GeneticCodeSCC"
        d["env_bits"] = self.env_bits
        d["state_bits"] = self.state_bits
        return d

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], arrays: dict[str, np.ndarray], **kwargs: Any
    ) -> GeneticCodeSCC:
        keys: list[int] = arrays["keys"].tolist()
        values: list[int] = arrays["values"].tolist()
        code = dict(zip(keys, values))
        return cls(
            code,
            env_bits=d.get("env_bits", 0),
            state_bits=d.get("state_bits", 0),
            seed=d.get("seed"),
            resp_bits=d.get("resp_bits", 1),
        )


class GeneticCodeGraph(GeneticCodeDict):
    """A GeneticCodeDict that builds an empirical behavioral transition graph
    at runtime and uses it to preserve epistatic state chains during crossover.

    Every ``__getitem__`` call records a directed edge from the *previous* key
    queried to the *current* key, incrementing a count for that edge.  This
    builds an empirical map of which code entries fire in sequence without
    any knowledge of env_bits or state_bits — making the implementation
    fully environment-agnostic.

    During crossover, after randomly assigning a top-level mapping to the
    child, a BFS traverses the *primary* parent's graph and pulls in
    transitionally-connected entries whose edge count is at or above
    ``edge_threshold``.  If the *secondary* parent has a strictly higher count
    for the same edge, its value is used instead (tie-break by strength).
    Entries already present in the child are never overwritten.

    The behavioral graph is **ephemeral** — it is not persisted in checkpoints
    and resets to empty each episode via ``reset_trajectory()``, which
    ``AutomatonBase.reset()`` calls automatically between generations.

    Args:
        code:           Initial mapping (as for GeneticCodeDict).
        seed:           Optional RNG seed.
        resp_bits:      Total output bit-width.
        edge_threshold: Minimum edge count for an edge to trigger a BFS pull
                        during crossover (default: 3).
    """

    def __init__(
        self,
        code: Mapping[int, int] | Sequence[int],
        seed: int | None = None,
        resp_bits: int = 1,
        edge_threshold: int = 3,
    ) -> None:
        super().__init__(code, seed=seed, resp_bits=resp_bits)
        self.edge_threshold = edge_threshold
        # Empirical transition graph: src_key → {dst_key → count}
        self._graph: dict[int, dict[int, int]] = {}
        self._last_key: int | None = None

    # ------------------------------------------------------------------
    # Runtime graph building
    # ------------------------------------------------------------------

    def __getitem__(self, key: int) -> int:
        value = super().__getitem__(key)
        last = self._last_key
        if last is not None:
            edges = self._graph.get(last)
            if edges is None:
                edges = {}
                self._graph[last] = edges
            edges[key] = edges.get(key, 0) + 1
        self._last_key = key
        return value

    def reset_trajectory(self) -> None:
        """Reset the last-key pointer.

        Clears only the trajectory cursor, not the accumulated graph — the
        graph is reset implicitly by creating a fresh instance in crossover().
        """
        self._last_key = None

    # ------------------------------------------------------------------
    # Crossover
    # ------------------------------------------------------------------

    def _bfs_pull(
        self,
        start_key: int,
        primary: GeneticCodeGraph,
        secondary: GeneticCodeGraph,
        child_code: dict[int, int],
        threshold: int,
    ) -> None:
        """BFS along primary's graph edges starting from *start_key*.

        For each reachable dst_key whose edge count meets *threshold* and that
        is not yet in *child_code*, a mapping is inserted.  If secondary has a
        strictly higher count for that specific edge, secondary's value is used
        (tie-break by behavioral strength); otherwise primary's value is used.
        """
        from collections import deque

        queue: deque[int] = deque([start_key])
        visited: set[int] = {start_key}

        while queue:
            current = queue.popleft()
            primary_edges = primary._graph.get(current, {})

            for dst_key, p_count in primary_edges.items():
                if dst_key in visited or dst_key in child_code:
                    continue
                if p_count < threshold:
                    continue

                # Tie-break: prefer secondary's value if its count is strictly
                # higher (stronger behavioral evidence for this transition).
                s_count = secondary._graph.get(current, {}).get(dst_key, 0)
                if s_count > p_count and dst_key in secondary._code:
                    child_code[dst_key] = secondary._code[dst_key]
                elif dst_key in primary._code:
                    child_code[dst_key] = primary._code[dst_key]
                else:
                    continue  # key absent from both _code dicts; skip

                visited.add(dst_key)
                queue.append(dst_key)

    def crossover(
        self, other: GeneticCode, mutation_rate: float = 0.01
    ) -> GeneticCodeGraph:
        """Graph-aware crossover with BFS pull and edge-strength tie-breaking.

        Algorithm
        ---------
        1. Collect all keys from both parents in random order.
        2. For each key not yet in the child, pick a primary parent (50/50 if
           both have the key, otherwise the sole owner).
        3. Insert that mapping, then BFS-pull transitionally-connected entries
           from the primary's graph (secondary value used on tie-break).
        4. Skip any key already claimed by a BFS pull — no overwrites.
        5. Apply per-entry point-mutation via geometric sampling.
        """
        assert isinstance(other, GeneticCodeGraph), (
            "GeneticCodeGraph can only crossover with another GeneticCodeGraph."
        )

        rng = self._rng
        threshold = self.edge_threshold

        all_keys = list(set(self._code) | set(other._code))
        rng.shuffle(all_keys)

        child_code: dict[int, int] = {}

        for key in all_keys:
            if key in child_code:
                # Already claimed by a BFS pull — do not overwrite.
                continue

            in_a = key in self._code
            in_b = key in other._code

            if in_a and in_b:
                primary, secondary = (
                    (self, other) if rng.random() < 0.5 else (other, self)
                )
            elif in_a:
                primary, secondary = self, other
            else:
                primary, secondary = other, self

            child_code[key] = primary._code[key]

            # Pull transitionally-connected entries from primary's graph,
            # using secondary as a tie-breaker for higher-count edges.
            self._bfs_pull(key, primary, secondary, child_code, threshold)

        # Per-entry point-mutation (geometric sampling — same as GeneticCodeDict).
        if mutation_rate > 0.0:
            resp_bits = self.resp_bits
            rnd = rng.random
            inv_log = 1.0 / log(1.0 - mutation_rate)
            keys = list(child_code)
            n = len(keys)
            i = int(log(1.0 - rnd()) * inv_log)
            while i < n:
                child_code[keys[i]] ^= 1 << rng.randrange(resp_bits)
                i += 1 + int(log(1.0 - rnd()) * inv_log)

        return GeneticCodeGraph(
            child_code,
            seed=rng.randint(0, 2**32 - 1),
            resp_bits=self.resp_bits,
            edge_threshold=self.edge_threshold,
        )

    # ------------------------------------------------------------------
    # Checkpoint interface
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["type"] = "GeneticCodeGraph"
        d["edge_threshold"] = self.edge_threshold
        return d

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        arrays: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> GeneticCodeGraph:
        keys: list[int] = arrays["keys"].tolist()
        values: list[int] = arrays["values"].tolist()
        code = dict(zip(keys, values))
        return cls(
            code,
            seed=d.get("seed"),
            resp_bits=d.get("resp_bits", 1),
            edge_threshold=d.get("edge_threshold", 3),
        )
