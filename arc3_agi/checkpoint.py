"""Checkpoint persistence infrastructure.

Each checkpoint is a two-file sidecar pair:
  <stem>.toml  -- human-readable metadata: class config, hyperparameters, fitness
                  history, coordinates, etc.
  <stem>.npz   -- compressed NumPy arrays: genetic code entries, energy grids, etc.

Both files are always written and read together via ``save``/``load``.
"""

from __future__ import annotations

import tomllib
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import tomli_w

SCHEMA_VERSION: int = 1


@dataclass
class CheckpointConfig:
    """Configuration for automatic checkpoint behaviour on a Population.

    Attributes:
        enabled:             Master switch. When False no files are ever written.
        base_dir:            Root directory under which run folders are created.
                             Each run gets its own sub-folder named by ISO timestamp.
        generation_interval: Write a checkpoint every N generations (calls to
                             ``evolve()``). 0 = disabled, 1 = every generation,
                             N = every N-th generation.
    """

    enabled: bool = True
    base_dir: Path = field(default_factory=lambda: Path("runs"))
    generation_interval: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "base_dir": str(self.base_dir),
            "generation_interval": self.generation_interval,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CheckpointConfig:
        return cls(
            enabled=d.get("enabled", True),
            base_dir=Path(d.get("base_dir", "runs")),
            generation_interval=d.get("generation_interval", 1),
        )


class Checkpointable:
    """Abstract mixin that adds save/load checkpoint support to a class.

    Subclasses must implement:
      * ``to_dict()``   -- return TOML-serialisable metadata (no large arrays).
      * ``to_arrays()`` -- return bulk numeric data as ``{name: np.ndarray}``.
      * ``from_dict()`` -- class-method that reconstructs an instance from both.

    The concrete ``save`` and ``load`` methods handle file I/O automatically.
    """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return all metadata as a plain dict suitable for TOML serialisation.

        Must not contain large numeric arrays -- those go in ``to_arrays()``.
        """

    @abstractmethod
    def to_arrays(self) -> dict[str, np.ndarray]:
        """Return bulk numeric data as a name → ndarray mapping for NPZ storage."""

    @classmethod
    @abstractmethod
    def from_dict(
        cls, d: dict[str, Any], arrays: dict[str, np.ndarray], **kwargs: Any
    ) -> Any:
        """Reconstruct an instance from the metadata dict and arrays dict.

        Subclasses that depend on external objects (e.g. an ``Environment``)
        should accept them as keyword arguments and validate consistency with
        the stored metadata before reconstructing.
        """

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Write ``<path>.toml`` (metadata) and ``<path>.npz`` (arrays).

        ``path`` should be provided without an extension; any extension is
        stripped before the two output files are written.
        """
        stem = path.with_suffix("")
        toml_path = stem.with_suffix(".toml")
        npz_path = stem.with_suffix(".npz")
        toml_path.parent.mkdir(parents=True, exist_ok=True)
        with toml_path.open("wb") as fh:
            tomli_w.dump(self.to_dict(), fh)
        arrays = self.to_arrays()
        if arrays:
            np.savez_compressed(npz_path, **arrays)  # type: ignore[arg-type]
        else:
            # Write an empty npz so load() always finds both files.
            np.savez_compressed(npz_path)

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> Any:
        """Read ``<path>.toml`` and ``<path>.npz`` and return a new instance.

        ``path`` may or may not include an extension; the stem is used for
        both files regardless.
        """
        stem = path.with_suffix("")
        toml_path = stem.with_suffix(".toml")
        npz_path = stem.with_suffix(".npz")
        with toml_path.open("rb") as fh:
            d = tomllib.load(fh)
        arrays: dict[str, np.ndarray] = dict(np.load(npz_path, allow_pickle=False))
        return cls.from_dict(d, arrays, **kwargs)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def genetic_code_from_dict(d: dict[str, Any], arrays: dict[str, np.ndarray]) -> Any:
    """Reconstruct a GeneticCode subclass from its metadata dict and arrays.

    Dispatches on ``d["type"]``.
    """
    # Import here to avoid circular imports at module level.
    from arc3_agi.genetic_code import GeneticCodeDict, GeneticCodeGraph, GeneticCodeList, GeneticCodeSCC

    gc_type = d.get("type")
    if gc_type == "GeneticCodeDict":
        return GeneticCodeDict.from_dict(d, arrays)
    if gc_type == "GeneticCodeList":
        return GeneticCodeList.from_dict(d, arrays)
    if gc_type == "GeneticCodeSCC":
        return GeneticCodeSCC.from_dict(d, arrays)
    if gc_type == "GeneticCodeGraph":
        return GeneticCodeGraph.from_dict(d, arrays)
    raise ValueError(f"Unknown GeneticCode type: {gc_type!r}")
