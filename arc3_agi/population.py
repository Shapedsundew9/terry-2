from __future__ import annotations

import datetime
from pathlib import Path
from random import randrange
from typing import Any

import numpy as np

from arc3_agi.automaton import AutomatonBase
from arc3_agi.checkpoint import SCHEMA_VERSION, Checkpointable, CheckpointConfig
from arc3_agi.environment import Environment
from arc3_agi.fingerprint import FingerprintConfig
from arc3_agi.genetic_code import GeneticCode


def _tournament_select(
    pool: list[AutomatonBase],
    selector: AutomatonBase | None,
    k: int,
) -> AutomatonBase:
    """Return a mate from ``pool`` using tournament selection.

    If ``k <= 1``, or if ``selector`` has no fingerprint, or the pool has only
    one candidate, a uniform-random draw is returned immediately (fast path).
    Otherwise ``k`` candidates are drawn with replacement and the one whose
    fingerprint has the lowest Hamming distance to ``selector``'s fingerprint
    is returned.
    """
    n = len(pool)
    if k <= 1 or selector is None or selector.fingerprint is None or n <= 1:
        return pool[randrange(n)]
    candidates = [pool[randrange(n)] for _ in range(k)]
    sel_fp = selector.fingerprint
    return min(
        candidates,
        key=lambda a: (
            sel_fp.hamming(a.fingerprint) if a.fingerprint is not None else sel_fp.bits
        ),
    )


class Population(Checkpointable):
    """Represents a population of automata for evolutionary processes."""

    def __init__(
        self,
        size: int,
        AutomatonClass: type[AutomatonBase],
        environment: Environment,
        checkpoint_config: CheckpointConfig | None = None,
        fingerprint_config: FingerprintConfig | None = None,
    ) -> None:
        self._automata_class = AutomatonClass
        self._fingerprint_config = fingerprint_config
        self.automata = [
            AutomatonClass(
                environment=environment, fingerprint_config=fingerprint_config
            )
            for _ in range(size)
        ]
        self.environment = environment
        self.tick_count: int = 0
        self.generation: int = 0
        self.fitness_history: list[dict[str, Any]] = []
        self._gen_start_time: datetime.datetime = datetime.datetime.now()
        self.checkpoint_config = (
            checkpoint_config if checkpoint_config is not None else CheckpointConfig()
        )
        # Tracks (parent1, parent2, child, p1_fitness_at_mating, p2_fitness_at_mating)
        # for the fingerprint update rule applied at the start of the next evolve().
        self._prev_pairings: list[
            tuple[AutomatonBase, AutomatonBase, AutomatonBase, float, float]
        ] = []

        if self.checkpoint_config.enabled:
            ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
            self._run_dir: Path | None = self.checkpoint_config.base_dir / ts
            self._run_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._run_dir = None

    def tick(self) -> None:
        """Perform a tick for all automata using batched environment observation.

        Automata whose ``is_active`` property returns False are skipped — they
        have exhausted their energy budget and would only waste CPU.  They
        remain in the population list and their accumulated fitness is used
        normally at the next ``evolve()`` call.
        """
        for automaton in self.automata:
            if automaton.is_active:
                automaton.tick()
        self.tick_count += 1

    def evolve(self) -> list[float]:
        """Evolve the population based on fitness, track history, and checkpoint."""
        self.automata.sort(key=lambda a: a.fitness, reverse=True)

        # ------------------------------------------------------------------
        # Fingerprint update rule — applied to survivors from the previous
        # generation based on the fitness of the offspring they produced.
        # Only runs when fingerprinting is active and pairings were recorded.
        # ------------------------------------------------------------------
        fp_cfg = self._fingerprint_config
        if fp_cfg is not None and self._prev_pairings:
            half = len(self.automata) // 2
            # The survivor cutoff is the lowest fitness among the top half.
            # Using this as the flip_toward threshold makes the criterion
            # population-rank-based ("did the child beat the median?") rather
            # than parent-relative ("did the child beat its own parents?").
            # The parent_avg baseline was designed to be fair but has the
            # unintended consequence that top performers can never satisfy it
            # (their children rarely outscore them), so their fingerprints
            # never converge even when they consistently produce viable offspring.
            survivor_cutoff = self.automata[half - 1].fitness
            survivor_ids = {id(a) for a in self.automata[:half] if a.fitness > 0.0}
            # Track which learner/teacher pairs have already been updated this
            # generation to enforce one-update-per-unique-partner.
            seen: set[tuple[int, int]] = set()
            for p1, p2, child, p1_fit, p2_fit in self._prev_pairings:
                # Only the lower-fitness parent (the "learner") updates its
                # fingerprint relative to the higher-fitness parent ("teacher").
                # This keeps high-fitness fingerprints as stable attractors —
                # they are never pulled toward lower-fitness mates.
                # On a fitness tie p1 is the learner by convention.
                if p1_fit <= p2_fit:
                    learner, teacher = p1, p2
                else:
                    learner, teacher = p2, p1
                if id(learner) not in survivor_ids:
                    continue
                if learner.fingerprint is None or teacher.fingerprint is None:
                    continue
                pair_key = (id(learner), id(teacher))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                child_fit = child.fitness
                if child_fit > survivor_cutoff:
                    learner.fingerprint.flip_toward(teacher.fingerprint)
                elif id(child) not in survivor_ids:
                    learner.fingerprint.flip_away(teacher.fingerprint)
                # child survived but didn't beat the cutoff → no change

        # Keep the top 50% and replace the rest with offspring.
        survivors = self.automata[: len(self.automata) // 2]
        # D: Only breed from survivors that have earned positive fitness.
        # Zero-fitness automatons are kept alive this generation but cannot
        # be parents, so the pathological sitting-still genotype is never
        # directly propagated — it must be re-created by crossover each time,
        # and with only fit parents that pressure diminishes rapidly.
        breeding_pool = [a for a in survivors if a.fitness > 0.0]
        if not breeding_pool:
            # Fallback for pathological early generations where no automaton moved.
            breeding_pool = survivors[: max(1, len(survivors) // 10)]

        k = fp_cfg.tournament_k if fp_cfg is not None else 1
        offspring = []
        new_pairings: list[
            tuple[AutomatonBase, AutomatonBase, AutomatonBase, float, float]
        ] = []
        for i in range(len(self.automata) // 2):
            parent1 = _tournament_select(breeding_pool, None, k)
            assert isinstance(parent1.genetic_code, GeneticCode)
            parent2 = _tournament_select(breeding_pool, parent1, k)
            assert isinstance(parent2.genetic_code, GeneticCode)
            child_genetic_code = parent1.genetic_code.crossover(parent2.genetic_code)
            child = self._automata_class(
                genetic_code=child_genetic_code, environment=self.environment
            )
            # Cross over and mutate the fingerprint if active.
            if (
                fp_cfg is not None
                and parent1.fingerprint is not None
                and parent2.fingerprint is not None
            ):
                child.fingerprint = parent1.fingerprint.crossover(parent2.fingerprint)
                child.fingerprint.mutate(fp_cfg.mutation_rate)
            new_pairings.append(
                (parent1, parent2, child, parent1.fitness, parent2.fitness)
            )
            offspring.append(child)

        fitnesses = [a.fitness for a in self.automata]

        # Record fitness statistics for this generation.
        self.generation += 1
        _now = datetime.datetime.now()
        duration_s = (_now - self._gen_start_time).total_seconds()
        self.fitness_history.append(
            {
                "generation": self.generation,
                "min_fitness": min(fitnesses),
                "max_fitness": max(fitnesses),
                "mean_fitness": sum(fitnesses) / len(fitnesses),
                "duration_s": duration_s,
                "fitnesses": fitnesses,
            }
        )

        self.automata[len(self.automata) // 2 :] = offspring
        self._prev_pairings = new_pairings
        for a in self.automata:
            a.reset()

        self._gen_start_time = _now

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
            **(
                {"fingerprint_config": self._fingerprint_config.to_dict()}
                if self._fingerprint_config is not None
                else {}
            ),
            "fitness_history": [
                {
                    "generation": e["generation"],
                    "min_fitness": e["min_fitness"],
                    "max_fitness": e["max_fitness"],
                    "mean_fitness": e["mean_fitness"],
                    "duration_s": e["duration_s"],
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
        fp_cfg_dict = d.get("fingerprint_config")
        fp_cfg = (
            FingerprintConfig.from_dict(fp_cfg_dict)
            if fp_cfg_dict is not None
            else None
        )

        # Build population without calling __init__ (no new automata spawned).
        pop = cls.__new__(cls)
        pop._automata_class = AutomatonClass
        pop._fingerprint_config = fp_cfg
        pop.environment = environment
        pop.tick_count = meta["tick_count"]
        pop.generation = meta["generation"]
        # Pairings cannot be serialised (object references); the first evolve()
        # after restore will silently skip the fingerprint update for that gap.
        pop._prev_pairings = []
        # Rebuild fitness_history: summary stats from TOML, full arrays from NPZ.
        history_meta = d.get("fitness_history", [])
        fh_arrays = arrays.get("fitness_history_fitnesses")
        pop.fitness_history = [
            {
                **entry,
                "duration_s": entry.get("duration_s", None),
                "fitnesses": (fh_arrays[i].tolist() if fh_arrays is not None else []),
            }
            for i, entry in enumerate(history_meta)
        ]
        pop.checkpoint_config = cfg
        pop._gen_start_time = datetime.datetime.now()

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
