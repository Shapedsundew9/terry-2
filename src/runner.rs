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
use crate::checkpoint::{load_population, ResumeConfig};
use crate::maze::Maze;
use crate::population::{GenerationStats, PopConfig, PopulationAutomaton, PopulationCore};

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
    pub maze_name: String,
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
            maze_name: "MazeRunnerMaze".into(),
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
    run_one_population_core::<crate::automaton::MazeAutomaton, _>(
        pop_id,
        maze.as_ref(),
        run_dir,
        &cfg.pop_config,
        cfg.base_population_seed,
        cfg.max_generations,
        &cfg.checkpoint,
        crate::checkpoint::save_population,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_one_population_core<A, SaveCheckpoint>(
    pop_id: usize,
    environment: &A::Environment,
    run_dir: &Path,
    pop_config: &PopConfig,
    base_population_seed: u64,
    max_generations: usize,
    checkpoint: &CheckpointConfig,
    save_checkpoint: SaveCheckpoint,
) -> Vec<GenerationStats>
where
    A: PopulationAutomaton,
    SaveCheckpoint: Fn(&PopulationCore<A>, &A::Environment, &Path) -> std::io::Result<()>,
{
    let pop_dir = run_dir.join(format!("pop_{pop_id}"));
    std::fs::create_dir_all(&pop_dir).expect("create pop dir");

    let seed = base_population_seed + pop_id as u64;
    let mut population = PopulationCore::<A>::new(environment, pop_config.clone(), seed);

    let mut history: Vec<GenerationStats> = Vec::with_capacity(max_generations);

    let t0 = Instant::now();
    for generation in 0..max_generations {
        population.run_generation(environment);
        let stats = population.evolve(environment);

        // Optional checkpoint.
        if checkpoint.enabled
            && checkpoint.generation_interval > 0
            && (generation + 1) % checkpoint.generation_interval == 0
        {
            let stem = pop_dir.join(format!("gen_{:06}", generation + 1));
            if let Err(error) = save_checkpoint(&population, environment, &stem) {
                eprintln!(
                    "[pop {pop_id}] checkpoint error at gen {}: {error}",
                    generation + 1
                );
            }
        }

        history.push(stats);
    }

    let elapsed = t0.elapsed().as_secs_f64();
    eprintln!(
        "[pop {pop_id}] done: {} gens in {elapsed:.1}s  \
         best_max={:.3}",
        max_generations,
        history
            .iter()
            .map(|s| s.max_fitness)
            .fold(f64::NEG_INFINITY, f64::max),
    );

    // Write fitness_history.json in the format Python's ingest_run expects.
    write_fitness_history(&pop_dir, pop_id, &history);

    history
}

/// Serialise `fitness_history.json` matching Python's `_worker_fn` output.
pub(crate) fn write_fitness_history(pop_dir: &Path, pop_id: usize, history: &[GenerationStats]) {
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

/// Resume one checkpointed population to `max_generations` total generations.
pub fn run_resumed_population(
    checkpoint: &Path,
    run_dir: &Path,
    cfg: &RunnerConfig,
) -> Result<Vec<GenerationStats>, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(run_dir)?;
    let pop_dir = run_dir.join("pop_0");
    std::fs::create_dir_all(&pop_dir)?;
    let maze = Maze::new(&cfg.maze_name, cfg.side_length_bits, cfg.maze_seed);
    let mut population = load_population(
        checkpoint,
        &maze,
        &ResumeConfig {
            ticks_per_restart: cfg.pop_config.ticks_per_restart,
            restarts_per_gen: cfg.pop_config.restarts_per_gen,
            checkpoint_interval: cfg.checkpoint.generation_interval,
            mutation_rate: cfg.pop_config.mutation_rate,
            seed: cfg.base_population_seed,
        },
    )?;
    if cfg.max_generations < population.generation {
        return Err(format!(
            "target generation {} is below checkpoint generation {}",
            cfg.max_generations, population.generation
        )
        .into());
    }

    // Schema-v1 checkpoints do not preserve RNG state or the maze itself.
    // Start the next episode deterministically against the supplied maze.
    for automaton in &mut population.automata {
        automaton.reset(&maze);
    }

    let source_generation = population.generation;
    let started = Instant::now();
    while population.generation < cfg.max_generations {
        population.run_generation(&maze);
        population.evolve(&maze);
        let generation = population.generation;
        if cfg.checkpoint.enabled
            && cfg.checkpoint.generation_interval > 0
            && generation % cfg.checkpoint.generation_interval == 0
        {
            let stem = pop_dir.join(format!("gen_{generation:06}"));
            crate::checkpoint::save_population(&population, &maze, &stem)?;
        }
    }

    eprintln!(
        "[pop 0] resumed generation {source_generation} -> {} in {:.1}s",
        population.generation,
        started.elapsed().as_secs_f64()
    );
    write_fitness_history(&pop_dir, 0, &population.fitness_history);
    Ok(population.fitness_history)
}

/// Launch `total_populations` populations concurrently using a Rayon pool.
///
/// Returns the final-generation stats for every completed population (in
/// population-id order).
pub fn run_pool(run_dir: &Path, cfg: &RunnerConfig) -> Vec<Vec<GenerationStats>> {
    std::fs::create_dir_all(run_dir).expect("create run dir");

    // Build maze once and share across threads.
    let maze = Arc::new(Maze::new(
        &cfg.maze_name,
        cfg.side_length_bits,
        cfg.maze_seed,
    ));

    println!(
        "  Maze: {}×{}   pop_size: {}   gens: {}   restarts/gen: {}   ticks/restart: {}",
        maze.width,
        maze.height,
        cfg.pop_config.size,
        cfg.max_generations,
        cfg.pop_config.restarts_per_gen,
        cfg.pop_config.ticks_per_restart,
    );

    run_population_pool::<crate::automaton::MazeAutomaton, _>(
        run_dir,
        maze,
        cfg.total_populations,
        cfg.max_parallel,
        cfg.max_generations,
        &cfg.pop_config,
        cfg.base_population_seed,
        &cfg.checkpoint,
        crate::checkpoint::save_population,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_population_pool<A, SaveCheckpoint>(
    run_dir: &Path,
    environment: Arc<A::Environment>,
    total_populations: usize,
    max_parallel: usize,
    max_generations: usize,
    pop_config: &PopConfig,
    base_population_seed: u64,
    checkpoint: &CheckpointConfig,
    save_checkpoint: SaveCheckpoint,
) -> Vec<Vec<GenerationStats>>
where
    A: PopulationAutomaton + Send,
    A::Environment: Send + Sync,
    SaveCheckpoint:
        Fn(&PopulationCore<A>, &A::Environment, &Path) -> std::io::Result<()> + Send + Sync,
{
    std::fs::create_dir_all(run_dir).expect("create run dir");

    let pool = ThreadPoolBuilder::new()
        .num_threads(max_parallel)
        .build()
        .expect("build rayon thread pool");

    let results: Vec<(usize, Vec<GenerationStats>)> = pool.install(|| {
        use rayon::prelude::*;
        (0..total_populations)
            .into_par_iter()
            .map(|pop_id| {
                let history = run_one_population_core::<A, _>(
                    pop_id,
                    environment.as_ref(),
                    run_dir,
                    pop_config,
                    base_population_seed,
                    max_generations,
                    checkpoint,
                    &save_checkpoint,
                );
                (pop_id, history)
            })
            .collect()
    });

    // Return in population-id order.
    let mut results = results;
    results.sort_by_key(|(id, _)| *id);
    results.into_iter().map(|(_, h)| h).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genetic_code::GeneticCodeConfig;
    use crate::population::Population;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_directory(label: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("terry-{label}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn resume_uses_target_generation_and_writes_complete_history() {
        let root = test_directory("resume");
        std::fs::create_dir_all(&root).unwrap();
        let maze = Maze::new("resume-maze", 4, 5);
        let pop_config = PopConfig {
            size: 4,
            state_bits: 4,
            ticks_per_restart: 4,
            restarts_per_gen: 1,
            checkpoint_interval: 0,
            mutation_rate: 0.01,
            genetic_code: GeneticCodeConfig::default(),
            fingerprint: None,
        };
        let mut source = Population::new(&maze, pop_config.clone(), 10);
        source.run_generation(&maze);
        source.evolve(&maze);
        let checkpoint = root.join("source");
        crate::checkpoint::save_population(&source, &maze, &checkpoint).unwrap();

        let config = RunnerConfig {
            total_populations: 1,
            max_parallel: 1,
            max_generations: 3,
            pop_config,
            base_population_seed: 99,
            maze_seed: 5,
            maze_name: "resume-maze".into(),
            side_length_bits: 4,
            checkpoint: CheckpointConfig {
                enabled: true,
                generation_interval: 3,
            },
        };
        let first_dir = root.join("first");
        let second_dir = root.join("second");
        let first = run_resumed_population(&checkpoint, &first_dir, &config).unwrap();
        let second = run_resumed_population(&checkpoint, &second_dir, &config).unwrap();

        assert_eq!(first.len(), 3);
        assert_eq!(first.last().unwrap().generation, 3);
        assert_eq!(
            first
                .iter()
                .map(|stats| &stats.fitnesses)
                .collect::<Vec<_>>(),
            second
                .iter()
                .map(|stats| &stats.fitnesses)
                .collect::<Vec<_>>()
        );
        assert!(first_dir.join("pop_0/gen_000003.toml").exists());
        let history: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(first_dir.join("pop_0/fitness_history.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(history["history"].as_array().unwrap().len(), 3);
        assert_eq!(history["history"][0]["generation"], 1);
        assert_eq!(history["history"][2]["generation"], 3);
        std::fs::remove_dir_all(root).ok();
    }
}
