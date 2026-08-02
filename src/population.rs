/// Population — evolutionary loop over a collection of `MazeAutomaton`s.
///
/// Mirrors Python's `Population` class: `run_generation` drives the tick
/// loop (with multiple restarts), and `evolve` performs selection, crossover,
/// and mutation to produce the next generation.
use std::time::Instant;

use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::automaton::MazeAutomaton;
use crate::fingerprint::{FingerprintConfig, SelectionFingerprint};
use crate::genetic_code::{GeneticCode, GeneticCodeConfig};
use crate::maze::Maze;

/// Per-generation fitness statistics.  Written to `fitness_history.json`.
#[derive(Clone, Debug)]
pub struct GenerationStats {
    pub generation: usize,
    pub min_fitness: f64,
    pub max_fitness: f64,
    pub mean_fitness: f64,
    pub duration_s: f64,
    /// Full per-automaton fitness vector for NPZ checkpoint.
    pub fitnesses: Vec<f64>,
}

/// Configuration parameters for a single population run.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct PopConfig {
    pub size: usize,
    pub state_bits: u8,
    pub ticks_per_restart: usize,
    pub restarts_per_gen: usize,
    pub checkpoint_interval: usize,
    pub mutation_rate: f64,
    pub genetic_code: GeneticCodeConfig,
    pub fingerprint: Option<FingerprintConfig>,
}

impl Default for PopConfig {
    fn default() -> Self {
        PopConfig {
            size: 100,
            state_bits: 4,
            ticks_per_restart: 100,
            restarts_per_gen: 20,
            checkpoint_interval: 0,
            mutation_rate: 0.01,
            genetic_code: GeneticCodeConfig::default(),
            fingerprint: None,
        }
    }
}

pub trait PopulationAutomaton: Sized {
    type Environment;

    fn new(
        environment: &Self::Environment,
        state_bits: u8,
        code_config: &GeneticCodeConfig,
        seed: u64,
    ) -> Result<Self, String>;
    fn with_code(
        genetic_code: GeneticCode,
        environment: &Self::Environment,
        state_bits: u8,
        seed: u64,
    ) -> Self;
    fn tick(&mut self, environment: &Self::Environment);
    fn reset(&mut self, environment: &Self::Environment);
    fn is_active(&self) -> bool;
    fn id(&self) -> u64;
    fn set_id(&mut self, id: u64);
    fn fitness(&self) -> f64;
    fn set_fitness(&mut self, fitness: f64);
    fn genetic_code(&self) -> &GeneticCode;
    fn fingerprint(&self) -> Option<&SelectionFingerprint>;
    fn fingerprint_mut(&mut self) -> &mut Option<SelectionFingerprint>;
}

impl PopulationAutomaton for MazeAutomaton {
    type Environment = Maze;

    fn new(
        maze: &Maze,
        state_bits: u8,
        code_config: &GeneticCodeConfig,
        seed: u64,
    ) -> Result<Self, String> {
        MazeAutomaton::new(maze, state_bits, code_config, seed)
    }

    fn with_code(genetic_code: GeneticCode, maze: &Maze, state_bits: u8, seed: u64) -> Self {
        MazeAutomaton::with_code(genetic_code, maze, state_bits, seed)
    }

    fn tick(&mut self, maze: &Maze) {
        MazeAutomaton::tick(self, maze);
    }

    fn reset(&mut self, maze: &Maze) {
        MazeAutomaton::reset(self, maze);
    }

    fn is_active(&self) -> bool {
        MazeAutomaton::is_active(self)
    }

    fn id(&self) -> u64 {
        self.id
    }

    fn set_id(&mut self, id: u64) {
        self.id = id;
    }

    fn fitness(&self) -> f64 {
        self.fitness
    }

    fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness;
    }

    fn genetic_code(&self) -> &GeneticCode {
        &self.genetic_code
    }

    fn fingerprint(&self) -> Option<&SelectionFingerprint> {
        self.fingerprint.as_ref()
    }

    fn fingerprint_mut(&mut self) -> &mut Option<SelectionFingerprint> {
        &mut self.fingerprint
    }
}

/// Environment-independent evolutionary loop over a population of automata.
pub struct PopulationCore<A: PopulationAutomaton> {
    pub automata: Vec<A>,
    pub generation: usize,
    pub tick_count: u64,
    pub fitness_history: Vec<GenerationStats>,
    rng: Xoshiro256PlusPlus,
    pub config: PopConfig,
    gen_start: Instant,
    next_individual_id: u64,
    previous_pairings: Vec<Pairing>,
}

pub type Population = PopulationCore<MazeAutomaton>;

#[derive(Clone, Debug)]
struct Pairing {
    parent1_id: u64,
    parent2_id: u64,
    child_id: u64,
    parent1_fitness: f64,
    parent2_fitness: f64,
    parent1_fingerprint: Option<SelectionFingerprint>,
    parent2_fingerprint: Option<SelectionFingerprint>,
}

impl<A: PopulationAutomaton> PopulationCore<A> {
    pub fn restore(
        mut automata: Vec<A>,
        generation: usize,
        tick_count: u64,
        fitness_history: Vec<GenerationStats>,
        config: PopConfig,
        seed: u64,
    ) -> Self {
        for (id, automaton) in automata.iter_mut().enumerate() {
            automaton.set_id(id as u64);
        }
        let next_individual_id = automata.len() as u64;
        Self {
            automata,
            generation,
            tick_count,
            fitness_history,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
            config,
            gen_start: Instant::now(),
            next_individual_id,
            previous_pairings: Vec::new(),
        }
    }

    /// Create a new population seeded from `seed`.
    pub fn new(environment: &A::Environment, config: PopConfig, seed: u64) -> Self {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        if let Some(fingerprint) = &config.fingerprint {
            fingerprint
                .validate()
                .expect("invalid fingerprint configuration");
        }
        let automata: Vec<A> = (0..config.size)
            .map(|id| {
                let mut automaton = A::new(
                    environment,
                    config.state_bits,
                    &config.genetic_code,
                    rng.next_u64(),
                )?;
                automaton.set_id(id as u64);
                *automaton.fingerprint_mut() = config
                    .fingerprint
                    .as_ref()
                    .map(|cfg| SelectionFingerprint::random(cfg.bits, &mut rng));
                Ok::<A, String>(automaton)
            })
            .collect::<Result<_, _>>()
            .expect("invalid genetic-code configuration");
        let next_individual_id = automata.len() as u64;

        PopulationCore {
            automata,
            generation: 0,
            tick_count: 0,
            fitness_history: Vec::new(),
            rng,
            config,
            gen_start: Instant::now(),
            next_individual_id,
            previous_pairings: Vec::new(),
        }
    }

    /// Run one full generation: `restarts_per_gen` independent episodes each
    /// of `ticks_per_restart` ticks.  Fitness is set to the mean across all
    /// restarts.
    pub fn run_generation(&mut self, environment: &A::Environment) {
        let n = self.automata.len();
        let mut fitness_acc = vec![0.0f64; n];

        for restart in 0..self.config.restarts_per_gen {
            if restart > 0 {
                // Reset automata (but not genetic codes) between restarts.
                for a in &mut self.automata {
                    a.reset(environment);
                }
            }

            for _ in 0..self.config.ticks_per_restart {
                for a in &mut self.automata {
                    a.tick(environment);
                }
                self.tick_count += 1;
            }

            for (i, a) in self.automata.iter().enumerate() {
                fitness_acc[i] += a.fitness();
            }
        }

        // Replace each automaton's fitness with the generation mean.
        let restarts = self.config.restarts_per_gen as f64;
        for (i, a) in self.automata.iter_mut().enumerate() {
            a.set_fitness(fitness_acc[i] / restarts);
        }
    }

    /// Evolve the population: select, crossover, mutate, and reset.
    ///
    /// Returns the full fitness vector (before offspring replace bottom half)
    /// for progress reporting.
    pub fn evolve(&mut self, environment: &A::Environment) -> GenerationStats {
        // Sort descending by fitness.
        self.automata
            .sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());

        let n = self.automata.len();
        let half = n / 2;

        self.update_fingerprints(half);

        // Capture fitness vector before replacement (for history).
        let fitnesses: Vec<f64> = self.automata.iter().map(A::fitness).collect();

        // Build breeding pool from survivors with positive fitness.
        let pool_indices: Vec<usize> = (0..half)
            .filter(|&i| self.automata[i].fitness() > 0.0)
            .collect();

        let pool_indices = if pool_indices.is_empty() {
            // Fallback: top 10 % of survivors (at least 1).
            let fallback = (half / 10).max(1);
            (0..fallback).collect()
        } else {
            pool_indices
        };

        let pool_len = pool_indices.len();
        let mutation_rate = self.config.mutation_rate;
        let tournament_k = self
            .config
            .fingerprint
            .as_ref()
            .map_or(1, |config| config.tournament_k);

        // Create offspring.
        let mut offspring: Vec<A> = Vec::with_capacity(half);
        let mut new_pairings = Vec::with_capacity(half);
        for _ in 0..half {
            let p1_idx = pool_indices[(self.rng.next_u64() as usize) % pool_len];
            let p2_idx = self.select_mate(&pool_indices, p1_idx, tournament_k);
            let child_code = self.automata[p1_idx]
                .genetic_code()
                .crossover(
                    self.automata[p2_idx].genetic_code(),
                    mutation_rate,
                    &mut self.rng,
                )
                .expect("incompatible genetic-code parents");
            let child_seed = self.rng.next_u64();
            let mut child =
                A::with_code(child_code, environment, self.config.state_bits, child_seed);
            child.set_id(self.next_individual_id);
            self.next_individual_id += 1;
            if let (Some(first), Some(second), Some(config)) = (
                self.automata[p1_idx].fingerprint(),
                self.automata[p2_idx].fingerprint(),
                self.config.fingerprint.as_ref(),
            ) {
                let mut fingerprint = first.crossover(second, &mut self.rng);
                fingerprint.mutate(config.mutation_rate, &mut self.rng);
                *child.fingerprint_mut() = Some(fingerprint);
            }
            new_pairings.push(Pairing {
                parent1_id: self.automata[p1_idx].id(),
                parent2_id: self.automata[p2_idx].id(),
                child_id: child.id(),
                parent1_fitness: self.automata[p1_idx].fitness(),
                parent2_fitness: self.automata[p2_idx].fitness(),
                parent1_fingerprint: self.automata[p1_idx].fingerprint().cloned(),
                parent2_fingerprint: self.automata[p2_idx].fingerprint().cloned(),
            });
            offspring.push(child);
        }

        // Record statistics.
        self.generation += 1;
        let duration_s = self.gen_start.elapsed().as_secs_f64();
        self.gen_start = Instant::now();

        let min_f = fitnesses.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_f = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_f = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;

        let stats = GenerationStats {
            generation: self.generation,
            min_fitness: min_f,
            max_fitness: max_f,
            mean_fitness: mean_f,
            duration_s,
            fitnesses: fitnesses.clone(),
        };
        self.fitness_history.push(stats.clone());

        // Replace bottom half with offspring, then reset all automata.
        for (i, child) in offspring.into_iter().enumerate() {
            self.automata[half + i] = child;
        }
        self.previous_pairings = new_pairings;
        for a in &mut self.automata {
            a.reset(environment);
        }

        stats
    }

    fn select_mate(&mut self, pool: &[usize], selector: usize, k: usize) -> usize {
        if k <= 1 || pool.len() <= 1 || self.automata[selector].fingerprint().is_none() {
            return pool[(self.rng.next_u64() as usize) % pool.len()];
        }
        let selector_fingerprint = self.automata[selector].fingerprint().unwrap();
        let mut best = pool[(self.rng.next_u64() as usize) % pool.len()];
        let mut best_distance = self.automata[best]
            .fingerprint()
            .map_or(u32::MAX, |candidate| {
                selector_fingerprint.hamming(candidate)
            });
        for _ in 1..k {
            let candidate = pool[(self.rng.next_u64() as usize) % pool.len()];
            let distance = self.automata[candidate]
                .fingerprint()
                .map_or(u32::MAX, |fingerprint| {
                    selector_fingerprint.hamming(fingerprint)
                });
            if distance < best_distance {
                best = candidate;
                best_distance = distance;
            }
        }
        best
    }

    fn update_fingerprints(&mut self, half: usize) {
        if self.config.fingerprint.is_none() || self.previous_pairings.is_empty() || half == 0 {
            return;
        }
        let survivor_cutoff = self.automata[half - 1].fitness();
        let survivor_ids: std::collections::HashSet<u64> = self.automata[..half]
            .iter()
            .filter(|automaton| automaton.fitness() > 0.0)
            .map(A::id)
            .collect();
        let fitness_by_id: std::collections::HashMap<u64, f64> = self
            .automata
            .iter()
            .map(|automaton| (automaton.id(), automaton.fitness()))
            .collect();
        let fingerprint_by_id: std::collections::HashMap<u64, SelectionFingerprint> = self
            .automata
            .iter()
            .filter_map(|automaton| {
                automaton
                    .fingerprint()
                    .cloned()
                    .map(|fingerprint| (automaton.id(), fingerprint))
            })
            .collect();
        let mut seen = std::collections::HashSet::new();

        for pairing in &self.previous_pairings {
            let (learner_id, teacher_id) = if pairing.parent1_fitness <= pairing.parent2_fitness {
                (pairing.parent1_id, pairing.parent2_id)
            } else {
                (pairing.parent2_id, pairing.parent1_id)
            };
            if !survivor_ids.contains(&learner_id) || !seen.insert((learner_id, teacher_id)) {
                continue;
            }
            let saved_teacher = if teacher_id == pairing.parent1_id {
                pairing.parent1_fingerprint.as_ref()
            } else {
                pairing.parent2_fingerprint.as_ref()
            };
            let Some(teacher) = fingerprint_by_id.get(&teacher_id).or(saved_teacher) else {
                continue;
            };
            let child_fitness = fitness_by_id.get(&pairing.child_id).copied();
            let child_survived = survivor_ids.contains(&pairing.child_id);
            let Some(learner) = self
                .automata
                .iter_mut()
                .find(|automaton| automaton.id() == learner_id)
                .and_then(|automaton| automaton.fingerprint_mut().as_mut())
            else {
                continue;
            };
            if child_fitness.is_some_and(|fitness| fitness > survivor_cutoff) {
                learner.flip_toward(teacher, &mut self.rng);
            } else if !child_survived {
                learner.flip_away(teacher, &mut self.rng);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genetic_code::GeneticCodeKind;

    #[test]
    fn fresh_populations_use_the_selected_genetic_code() {
        let maze = Maze::new("model-routing", 4, 3);
        for kind in [
            GeneticCodeKind::Tsetlin,
            GeneticCodeKind::Dict,
            GeneticCodeKind::List,
        ] {
            let config = PopConfig {
                size: 4,
                state_bits: 4,
                ticks_per_restart: 2,
                restarts_per_gen: 1,
                checkpoint_interval: 0,
                mutation_rate: 0.01,
                genetic_code: GeneticCodeConfig {
                    kind,
                    ..GeneticCodeConfig::default()
                },
                fingerprint: None,
            };
            let mut population = Population::new(&maze, config, 10);
            assert!(population
                .automata
                .iter()
                .all(|automaton| automaton.genetic_code.kind() == kind));
            population.run_generation(&maze);
            population.evolve(&maze);
            assert!(population
                .automata
                .iter()
                .all(|automaton| automaton.genetic_code.kind() == kind));
        }
    }
}
