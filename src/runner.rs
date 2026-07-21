/// Parallel population runner.
///
/// Uses a Rayon thread pool sized to `max_parallel` to run
/// `total_populations` independent populations concurrently.  Each population
/// writes a `fitness_history.json` file on completion.
///
/// The `fitness_history.json` format matches Python's `_worker_fn` exactly so
/// that `ExperimentStore.ingest_run` can consume Rust-generated output without
/// modification.
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use rayon::ThreadPoolBuilder;
use serde_json::json;

use crate::checkpoint::CheckpointConfig;
use crate::maze::Maze;
use crate::population::{GenerationStats, PopConfig, Population};

/// Parameters for the pool runner.
#[derive(Clone, Debug)]
pub struct RunnerConfig {
    pub total_populations: usize,
    pub max_parallel: usize,
    pub max_generations: usize,
    pub pop_config: PopConfig,
    /// Seed for the first population; each pop i gets `base_seed + i`.
    pub base_population_seed: u64,
    pub maze_seed: u64,
    pub side_length_bits: u8,
    pub checkpoint: CheckpointConfig,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        RunnerConfig {
            total_populations: 100,
            max_parallel: 12,
            max_generations: 10_000,
            pop_config: PopConfig::default(),
            base_population_seed: 0,
            maze_seed: 42,
            side_length_bits: 6,
            checkpoint: CheckpointConfig::default(),
        }
    }
}

/// Run a single population to completion and write `fitness_history.json`.
///
/// Returns the final `GenerationStats` for every generation.
pub fn run_one_population(
    pop_id: usize,
    maze: Arc<Maze>,
    run_dir: &Path,
    cfg: &RunnerConfig,
) -> Vec<GenerationStats> {
    let pop_dir = run_dir.join(format!("pop_{pop_id}"));
    std::fs::create_dir_all(&pop_dir).expect("create pop dir");

    let seed = cfg.base_population_seed + pop_id as u64;
    let mut population = Population::new(&maze, cfg.pop_config.clone(), seed);

    let ckpt_cfg = &cfg.checkpoint;
    let mut history: Vec<GenerationStats> = Vec::with_capacity(cfg.max_generations);

    let t0 = Instant::now();
    for gen in 0..cfg.max_generations {
        population.run_generation(&maze);
        let stats = population.evolve(&maze);

        // Optional checkpoint.
        if ckpt_cfg.enabled
            && ckpt_cfg.generation_interval > 0
            && (gen + 1) % ckpt_cfg.generation_interval == 0
        {
            let stem = pop_dir.join(format!("gen_{:06}", gen + 1));
            if let Err(e) = crate::checkpoint::save_population(&population, &maze, &stem) {
                eprintln!("[pop {pop_id}] checkpoint error at gen {}: {e}", gen + 1);
            }
        }

        history.push(stats);
    }

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "[pop {pop_id}] done: {} gens in {elapsed:.1}s  \
         best_max={:.3}",
        cfg.max_generations,
        history.iter().map(|s| s.max_fitness).fold(f64::NEG_INFINITY, f64::max),
    );

    // Write fitness_history.json in the format Python's ingest_run expects.
    write_fitness_history(&pop_dir, pop_id, &history);

    history
}

/// Serialise `fitness_history.json` matching Python's `_worker_fn` output.
fn write_fitness_history(pop_dir: &Path, pop_id: usize, history: &[GenerationStats]) {
    let records: Vec<serde_json::Value> = history
        .iter()
        .map(|s| {
            json!({
                "generation": s.generation,
                "min_fitness": s.min_fitness,
                "max_fitness": s.max_fitness,
                "mean_fitness": s.mean_fitness,
                "duration_s": s.duration_s,
            })
        })
        .collect();

    let doc = json!({
        "pop_id": pop_id,
        "history": records,
    });

    let path = pop_dir.join("fitness_history.json");
    let bytes = serde_json::to_vec_pretty(&doc).expect("json serialise");
    std::fs::write(&path, &bytes).expect("write fitness_history.json");
}

/// Launch `total_populations` populations concurrently using a Rayon pool.
///
/// Returns the final-generation stats for every completed population (in
/// population-id order).
pub fn run_pool(
    run_dir: &Path,
    cfg: &RunnerConfig,
) -> Vec<Vec<GenerationStats>> {
    std::fs::create_dir_all(run_dir).expect("create run dir");

    // Build maze once and share across threads.
    let maze = Arc::new(Maze::new("MazeRunnerMaze", cfg.side_length_bits, cfg.maze_seed));

    println!(
        "  Maze: {}×{}   pop_size: {}   gens: {}   restarts/gen: {}   ticks/restart: {}",
        maze.width,
        maze.height,
        cfg.pop_config.size,
        cfg.max_generations,
        cfg.pop_config.restarts_per_gen,
        cfg.pop_config.ticks_per_restart,
    );

    let pool = ThreadPoolBuilder::new()
        .num_threads(cfg.max_parallel)
        .build()
        .expect("build rayon thread pool");

    let results: Vec<(usize, Vec<GenerationStats>)> = pool.install(|| {
        use rayon::prelude::*;
        (0..cfg.total_populations)
            .into_par_iter()
            .map(|pop_id| {
                let history = run_one_population(pop_id, Arc::clone(&maze), run_dir, cfg);
                (pop_id, history)
            })
            .collect()
    });

    // Return in population-id order.
    let mut results = results;
    results.sort_by_key(|(id, _)| *id);
    results.into_iter().map(|(_, h)| h).collect()
}
