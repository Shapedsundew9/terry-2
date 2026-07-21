/// Population — evolutionary loop over a collection of `MazeAutomaton`s.
///
/// Mirrors Python's `Population` class: `run_generation` drives the tick
/// loop (with multiple restarts), and `evolve` performs selection, crossover,
/// and mutation to produce the next generation.
use std::time::Instant;

use rand::{RngCore, SeedableRng};
use rand::rngs::StdRng;

use crate::automaton::MazeAutomaton;
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
    pub code_type: String, // "GeneticCodeDict" | "GeneticCodeList"
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
            code_type: "GeneticCodeDict".into(),
        }
    }
}

/// A population of `MazeAutomaton`s evolving on a shared maze.
pub struct Population {
    pub automata: Vec<MazeAutomaton>,
    pub generation: usize,
    pub tick_count: u64,
    pub fitness_history: Vec<GenerationStats>,
    rng: StdRng,
    pub config: PopConfig,
    gen_start: Instant,
}

impl Population {
    /// Create a new population seeded from `seed`.
    pub fn new(maze: &Maze, config: PopConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let automata: Vec<MazeAutomaton> = (0..config.size)
            .map(|_| MazeAutomaton::new(maze, config.state_bits, rng.next_u64()))
            .collect();

        Population {
            automata,
            generation: 0,
            tick_count: 0,
            fitness_history: Vec::new(),
            rng,
            config,
            gen_start: Instant::now(),
        }
    }

    /// Run one full generation: `restarts_per_gen` independent episodes each
    /// of `ticks_per_restart` ticks.  Fitness is set to the mean across all
    /// restarts.
    pub fn run_generation(&mut self, maze: &Maze) {
        let n = self.automata.len();
        let mut fitness_acc = vec![0.0f64; n];

        for restart in 0..self.config.restarts_per_gen {
            if restart > 0 {
                // Reset automata (but not genetic codes) between restarts.
                for a in &mut self.automata {
                    a.reset(maze);
                }
            }

            for _ in 0..self.config.ticks_per_restart {
                for a in &mut self.automata {
                    if a.is_active() {
                        a.tick(maze);
                    }
                }
                self.tick_count += 1;
            }

            for (i, a) in self.automata.iter().enumerate() {
                fitness_acc[i] += a.fitness;
            }
        }

        // Replace each automaton's fitness with the generation mean.
        let restarts = self.config.restarts_per_gen as f64;
        for (i, a) in self.automata.iter_mut().enumerate() {
            a.fitness = fitness_acc[i] / restarts;
        }
    }

    /// Evolve the population: select, crossover, mutate, and reset.
    ///
    /// Returns the full fitness vector (before offspring replace bottom half)
    /// for progress reporting.
    pub fn evolve(&mut self, maze: &Maze) -> GenerationStats {
        // Sort descending by fitness.
        self.automata.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        let n = self.automata.len();
        let half = n / 2;

        // Capture fitness vector before replacement (for history).
        let fitnesses: Vec<f64> = self.automata.iter().map(|a| a.fitness).collect();

        // Build breeding pool from survivors with positive fitness.
        let pool_indices: Vec<usize> = (0..half)
            .filter(|&i| self.automata[i].fitness > 0.0)
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

        // Create offspring.
        let mut offspring: Vec<MazeAutomaton> = Vec::with_capacity(half);
        for _ in 0..half {
            let p1_idx = pool_indices[(self.rng.next_u64() as usize) % pool_len];
            let p2_idx = pool_indices[(self.rng.next_u64() as usize) % pool_len];
            // Clone p2's genetic code so we can borrow p1 and self.rng
            // independently (different struct fields — Rust allows this).
            let p2_code = self.automata[p2_idx].genetic_code.clone_box();
            let child_code = self.automata[p1_idx]
                .genetic_code
                .crossover(p2_code.as_ref(), mutation_rate, &mut self.rng);
            let child_seed = self.rng.next_u64();
            offspring.push(MazeAutomaton::with_code(
                child_code,
                maze,
                self.config.state_bits,
                child_seed,
            ));
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
        for a in &mut self.automata {
            a.reset(maze);
        }

        stats
    }
}
