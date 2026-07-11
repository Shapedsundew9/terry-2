"""Compatibility fingerprint for evolved mate preference.

Each automaton carries a ``SelectionFingerprint`` — a heritable integer bit
string whose similarity to a potential mate's fingerprint biases partner
selection via tournament sampling.  Over generations the fingerprint acquires
meaning through a flip-toward / flip-away update rule driven by offspring
fitness, causing the population to self-organise into compatibility clusters
without any domain-specific knowledge.

See ``docs/evolved_mate_preference_trait.md`` for the full design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any


@dataclass
class FingerprintConfig:
    """Configuration for the selection fingerprint mechanism.

    Attributes:
        bits:         Length of the fingerprint bit string. Default 32.
        tournament_k: Number of candidates drawn when selecting a mate.
                      k=1 means pure random selection (fast path — no
                      Hamming computation performed).  k>=2 enables
                      fingerprint-biased selection.
        mutation_rate: Independent per-bit mutation probability applied to
                      the fingerprint on each inheritance event.  Should
                      generally be set higher than the genome mutation rate
                      so the compatibility signal remains responsive.
    """

    bits: int = 32
    tournament_k: int = 1
    mutation_rate: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "tournament_k": self.tournament_k,
            "mutation_rate": self.mutation_rate,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FingerprintConfig:
        return cls(
            bits=int(d.get("bits", 32)),
            tournament_k=int(d.get("tournament_k", 1)),
            mutation_rate=float(d.get("mutation_rate", 0.01)),
        )


class SelectionFingerprint:
    """A heritable integer bit string used to bias mate selection.

    Similarity is measured by Hamming distance (``int.bit_count()`` on the
    XOR of two fingerprint values).  All mutation and update operations work
    on plain integers, avoiding floating-point arithmetic entirely except for
    the stochastic mutation rate comparison (``rng.random() < rate``).

    Args:
        bits:  Length of the fingerprint in bits.  Immutable after creation.
        rng:   Random instance used for all stochastic operations.  The
               automaton passes its own ``rng`` here so seeding is consistent.
        value: Initial integer value.  If None a random value is drawn from
               ``rng``.  Must satisfy ``0 <= value < 2**bits``.
    """

    __slots__ = ("value", "bits", "_mask", "_rng")

    def __init__(self, bits: int, rng: Random, value: int | None = None) -> None:
        self.bits: int = bits
        self._mask: int = (1 << bits) - 1
        self._rng: Random = rng
        self.value: int = (
            rng.getrandbits(bits) if value is None else (value & self._mask)
        )

    # ------------------------------------------------------------------
    # Core metric
    # ------------------------------------------------------------------

    def hamming(self, other: SelectionFingerprint) -> int:
        """Return the Hamming distance between this fingerprint and ``other``."""
        return (self.value ^ other.value).bit_count()

    # ------------------------------------------------------------------
    # Reproduction operations
    # ------------------------------------------------------------------

    def crossover(self, other: SelectionFingerprint) -> SelectionFingerprint:
        """Return a new fingerprint via uniform crossover.

        For each bit position the child independently inherits from ``self``
        or ``other`` with equal probability.  This guarantees:

        * Bits set in *both* parents are always inherited (AND preserved).
        * Bits set in *neither* parent are never inherited.
        * Each differing bit is drawn from ``self`` or ``other`` with
          probability 0.5, so the child is always "between" its parents in
          Hamming space at expected distance ``hamming(P1, P2) / 2`` from
          each parent.

        The child receives a fresh ``Random`` instance seeded from this
        fingerprint's own RNG, ensuring reproducibility under a fixed seed.
        """
        full_mask = (1 << self.bits) - 1
        sel = self._rng.getrandbits(self.bits)  # 1 → take from self, 0 → from other
        child_value = (self.value & sel) | (other.value & (full_mask ^ sel))
        child_rng = Random(self._rng.randrange(2**32))
        return SelectionFingerprint(self.bits, child_rng, value=child_value)

    def mutate(self, mutation_rate: float) -> None:
        """Flip each bit independently with probability ``mutation_rate``.

        In-place operation.  For the typical fingerprint length of 32 bits
        and low mutation rates this is 32 lightweight probability tests —
        negligible overhead compared with a genome crossover.
        """
        if mutation_rate <= 0.0:
            return
        rnd = self._rng
        if rnd.random() < mutation_rate:
            self.value ^= 1 << rnd.randrange(self.bits)

    # ------------------------------------------------------------------
    # Update rule (flip-toward / flip-away)
    # ------------------------------------------------------------------

    def flip_toward(self, other: SelectionFingerprint) -> None:
        """Flip one differing bit to match ``other`` (Hamming decreases by 1).

        If the fingerprints are already identical this is a no-op.
        """
        diff = self.value ^ other.value
        pos = self._random_set_bit(diff)
        if pos is not None:
            self.value ^= 1 << pos

    def flip_away(self, other: SelectionFingerprint) -> None:
        """Flip one matching bit to differ from ``other`` (Hamming increases by 1).

        If the fingerprints are already complementary (fully inverted) this
        is a no-op.
        """
        same = (~(self.value ^ other.value)) & self._mask
        pos = self._random_set_bit(same)
        if pos is not None:
            self.value ^= 1 << pos

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _random_set_bit(self, mask: int) -> int | None:
        """Return a uniformly random set-bit position within ``mask``.

        Returns None if ``mask`` is zero.  Uses integer popcount to count
        candidates then scans to the chosen one — purely integer arithmetic.
        """
        count = mask.bit_count()
        if count == 0:
            return None
        target = self._rng.randrange(count)
        pos = 0
        seen = 0
        while True:
            if (mask >> pos) & 1:
                if seen == target:
                    return pos
                seen += 1
            pos += 1

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"SelectionFingerprint(bits={self.bits}, value={self.value:#010x})"
