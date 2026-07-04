from __future__ import annotations

import datetime
from pathlib import Path
from random import randrange
from typing import Any

import numpy as np

from arc3_agi.automaton import AutomatonBase
from arc3_agi.checkpoint import SCHEMA_VERSION, Checkpointable, CheckpointConfig
from arc3_agi.environment import Environment
from arc3_agi.genetic_code import GeneticCode


class Population(Checkpointable):
    """Represents a population of automata for evolutionary processes."""

    def __init__(
        self,
        size: int,
        AutomatonClass: type[AutomatonBase],
        environment: Environment,
        checkpoint_config: CheckpointConfig | None = None,
    ) -> None:
        self._automata_class = AutomatonClass
        self.automata = [AutomatonClass(environment=environment) for _ in range(size)]
        self.environment = environment
        self.tick_count: int = 0
        self.generation: int = 0
        self.fitness_history: list[dict[str, Any]] = []
        self.checkpoint_config = (
            checkpoint_config if checkpoint_config is not None else CheckpointConfig()
        )

        if self.checkpoint_config.enabled:
            ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
            self._run_dir: Path | None = self.checkpoint_config.base_dir / ts
            self._run_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._run_dir = None

    def tick(self) -> None:
        """Perform a tick for all automata using batched environment observation."""
        for automaton in self.automata:
            automaton.tick()
        self.tick_count += 1

    def evolve(self) -> list[float]:
        """Evolve the population based on fitness, track history, and checkpoint."""
        self.automata.sort(key=lambda a: a.fitness, reverse=True)
        # Keep the top 50% and replace the rest with offspring.
        survivors = self.automata[: len(self.automata) // 2]
        offspring = []
        for i in range(len(self.automata) // 2):
            parent1 = survivors[randrange(len(survivors))]
            assert isinstance(parent1.genetic_code, GeneticCode)
            parent2 = survivors[randrange(len(survivors))]
            assert isinstance(parent2.genetic_code, GeneticCode)
            child_genetic_code = parent1.genetic_code.crossover(parent2.genetic_code)
            child = self._automata_class(
                genetic_code=child_genetic_code, environment=self.environment
            )
            offspring.append(child)

        fitnesses = [a.fitness for a in self.automata]

        # Record fitness statistics for this generation.
        self.generation += 1
        self.fitness_history.append(
            {
                "generation": self.generation,
                "min_fitness": min(fitnesses),
                "max_fitness": max(fitnesses),
                "mean_fitness": sum(fitnesses) / len(fitnesses),
                "fitnesses": fitnesses,
            }
        )

        self.automata[len(self.automata) // 2 :] = offspring
        for a in self.automata:
            a.reset()

        if (
            self.checkpoint_config.enabled
            and self.checkpoint_config.generation_interval > 0
        ):
            if self.generation % self.checkpoint_config.generation_interval == 0:
                self._save_checkpoint(f"gen_{self.generation:06d}")

        return fitnesses

    # ------------------------------------------------------------------
    # Internal checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self, stem: str) -> None:
        """Write a checkpoint pair (TOML + NPZ) to the run directory."""
        if self._run_dir is not None:
            self.save(self._run_dir / stem)

    # ------------------------------------------------------------------
    # Checkpoint interface
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": {
                "class": "Population",
                "schema_version": SCHEMA_VERSION,
                "generation": self.generation,
                "tick_count": self.tick_count,
                "automaton_class": self._automata_class.__name__,
            },
            "environment": {
                "class": type(self.environment).__name__,
                "name": self.environment.name,
            },
            "config": self.checkpoint_config.to_dict(),
            "fitness_history": [
                {
                    "generation": e["generation"],
                    "min_fitness": e["min_fitness"],
                    "max_fitness": e["max_fitness"],
                    "mean_fitness": e["mean_fitness"],
                }
                for e in self.fitness_history
            ],
            "automata": [
                a.to_dict()["automaton"] | {"genetic_code": a.to_dict()["genetic_code"]}
                for a in self.automata
            ],
        }

    def to_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        for i, automaton in enumerate(self.automata):
            prefix = f"automaton_{i}_"
            for k, v in automaton.to_arrays().items():
                arrays[f"{prefix}{k}"] = v
        if self.fitness_history:
            arrays["fitness_history_fitnesses"] = np.array(
                [e["fitnesses"] for e in self.fitness_history], dtype=np.float64
            )
        return arrays

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        arrays: dict[str, np.ndarray],
        **kwargs: Any,
    ) -> Population:
        environment: Environment = kwargs["environment"]
        AutomatonClass: type[AutomatonBase] = kwargs["AutomatonClass"]

        env_info = d["environment"]
        if (
            env_info["class"] != type(environment).__name__
            or env_info["name"] != environment.name
        ):
            raise ValueError(
                f"Environment mismatch: checkpoint has "
                f"{env_info['class']}/{env_info['name']!r} but received "
                f"{type(environment).__name__}/{environment.name!r}."
            )

        cfg = CheckpointConfig.from_dict(d.get("config", {}))
        meta = d["meta"]

        # Build population without calling __init__ (no new automata spawned).
        pop = cls.__new__(cls)
        pop._automata_class = AutomatonClass
        pop.environment = environment
        pop.tick_count = meta["tick_count"]
        pop.generation = meta["generation"]
        # Rebuild fitness_history: summary stats from TOML, full arrays from NPZ.
        history_meta = d.get("fitness_history", [])
        fh_arrays = arrays.get("fitness_history_fitnesses")
        pop.fitness_history = [
            {
                **entry,
                "fitnesses": (fh_arrays[i].tolist() if fh_arrays is not None else []),
            }
            for i, entry in enumerate(history_meta)
        ]
        pop.checkpoint_config = cfg

        # Create a fresh run dir for the resumed run.
        if cfg.enabled:
            ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
            pop._run_dir = cfg.base_dir / ts
            pop._run_dir.mkdir(parents=True, exist_ok=True)
        else:
            pop._run_dir = None

        # Reconstruct automata.
        automata_dicts = d.get("automata", [])
        pop.automata = []
        for i, a_meta in enumerate(automata_dicts):
            prefix = f"automaton_{i}_"
            a_arrays = {
                k[len(prefix) :]: v for k, v in arrays.items() if k.startswith(prefix)
            }
            # Wrap in the per-automaton dict shape expected by from_dict.
            a_full = {
                "meta": {
                    "class": AutomatonClass.__name__,
                    "schema_version": SCHEMA_VERSION,
                },
                "environment": env_info,
                "automaton": {k: v for k, v in a_meta.items() if k != "genetic_code"},
                "genetic_code": a_meta["genetic_code"],
            }
            pop.automata.append(
                AutomatonClass.from_dict(a_full, a_arrays, environment=environment)
            )

        return pop
