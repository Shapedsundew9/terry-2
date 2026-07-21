/// Maze-runner CLI entry point.
///
/// Run with `--help` to see all options.  Each invocation runs one
/// experiment (one pool of populations) and records results to PostgreSQL.
///
/// Example:
///   maze-runner --name baseline-state4 --state-bits 4
///   maze-runner --name baseline-state3 --state-bits 3 --populations 100
mod maze;
mod genetic_code;
mod automaton;
mod population;
mod runner;
mod checkpoint;
mod experiment;

use std::path::PathBuf;

use chrono::Utc;
use clap::Parser;
use rand::RngCore;
use serde_json::json;

use crate::checkpoint::CheckpointConfig;
use crate::experiment::{resolve_database_url, ExperimentStore};
use crate::population::PopConfig;
use crate::runner::{RunnerConfig, run_pool};

#[derive(Parser, Debug)]
#[command(
    name = "maze-runner",
    about = "High-performance maze evolution runner (Rust)",
    long_about = None,
)]
struct Cli {
    /// Experiment name (must be unique in the database).
    #[arg(long)]
    name: String,

    /// Free-text description stored in the database.
    #[arg(long, default_value = "")]
    description: String,

    /// Automaton state bits.
    #[arg(long, default_value_t = 4)]
    state_bits: u8,

    /// Total number of populations to run.
    #[arg(long, default_value_t = 100)]
    populations: usize,

    /// Maximum concurrent populations (Rayon thread pool size).
    #[arg(long, default_value_t = 12)]
    parallel: usize,

    /// Generations per population.
    #[arg(long, default_value_t = 10_000)]
    generations: usize,

    /// Ticks per restart.
    #[arg(long, default_value_t = 100)]
    ticks: usize,

    /// Restarts per generation.
    #[arg(long, default_value_t = 20)]
    restarts: usize,

    /// Automata per population.
    #[arg(long, default_value_t = 100)]
    pop_size: usize,

    /// Maze seed.
    #[arg(long, default_value_t = 42)]
    maze_seed: u64,

    /// Base population seed (pop i uses seed + i).
    #[arg(long, default_value_t = 0)]
    pop_seed: u64,

    /// Write a checkpoint every N generations (0 = only at the end).
    #[arg(long, default_value_t = 0)]
    checkpoint_interval: usize,

    /// Maze grid side-length bits (grid = 2^N × 2^N).
    #[arg(long, default_value_t = 6)]
    side_length_bits: u8,

    /// Root directory for checkpoint output.
    #[arg(long, default_value = "runs")]
    base_dir: PathBuf,

    /// PostgreSQL connection URL (overrides DATABASE_URL env var).
    #[arg(long)]
    database_url: Option<String>,

    /// Genetic code type: "dict" or "list".
    #[arg(long, default_value = "dict")]
    code_type: String,

    /// Per-bit mutation rate for crossover.
    #[arg(long, default_value_t = 0.01)]
    mutation_rate: f64,
}

fn generate_run_id() -> String {
    let ts = Utc::now().format("%Y%m%dT%H%M%S");
    let mut rng = rand::rng();
    let hex: u32 = (rng.next_u32()) & 0x00FFFFFF;
    format!("{ts}_{hex:06x}")
}

fn run_experiment(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let db_url = resolve_database_url(cli.database_url.as_deref());
    let run_id = generate_run_id();

    let code_type = match cli.code_type.as_str() {
        "list" => "GeneticCodeList",
        _ => "GeneticCodeDict",
    }
    .to_string();

    let checkpoint_cfg = if cli.checkpoint_interval > 0 {
        CheckpointConfig {
            enabled: true,
            generation_interval: cli.checkpoint_interval,
        }
    } else {
        // Write one checkpoint at the very end.
        CheckpointConfig {
            enabled: true,
            generation_interval: cli.generations,
        }
    };

    let pop_config = PopConfig {
        size: cli.pop_size,
        state_bits: cli.state_bits,
        ticks_per_restart: cli.ticks,
        restarts_per_gen: cli.restarts,
        checkpoint_interval: cli.checkpoint_interval,
        mutation_rate: cli.mutation_rate,
        code_type,
    };

    let runner_config = RunnerConfig {
        total_populations: cli.populations,
        max_parallel: cli.parallel,
        max_generations: cli.generations,
        pop_config,
        base_population_seed: cli.pop_seed,
        maze_seed: cli.maze_seed,
        side_length_bits: cli.side_length_bits,
        checkpoint: checkpoint_cfg,
    };

    let params = json!({
        "total_populations": cli.populations,
        "max_parallel": cli.parallel,
        "max_generations": cli.generations,
        "ticks_per_restart": cli.ticks,
        "restarts_per_gen": cli.restarts,
        "population_size": cli.pop_size,
        "side_length_bits": cli.side_length_bits,
        "maze_seed": cli.maze_seed,
        "population_seed": cli.pop_seed,
        "checkpoint_interval": cli.checkpoint_interval,
        "mutation_rate": cli.mutation_rate,
        "automaton_params": { "state_bits": cli.state_bits },
        "code_type": cli.code_type,
    });

    // --- Claim the experiment ---
    let mut store = ExperimentStore::connect(&db_url)?;
    let (experiment_id, already_done) = store.claim_experiment(
        &cli.name,
        &cli.description,
        &run_id,
        &params,
    )?;

    if already_done {
        println!(
            "\nExperiment '{}' already completed → id={experiment_id}; skipping.",
            cli.name
        );
        return Ok(());
    }

    store.mark_running(experiment_id)?;
    // Drop the store to release the connection while the pool runs.
    drop(store);

    println!(
        "\nMaze Runner (Rust) — '{}'\n  {} total × {} parallel × {} gens\n  Run ID: {run_id}  dir: {}",
        cli.name,
        cli.populations,
        cli.parallel,
        cli.generations,
        cli.base_dir.join(&run_id).display(),
    );

    let run_dir = cli.base_dir.join(&run_id);

    // --- Run the pool ---
    let result = std::panic::catch_unwind(|| {
        run_pool(&run_dir, &runner_config)
    });

    match result {
        Ok(_histories) => {
            // --- Ingest results ---
            let mut store = ExperimentStore::connect(&db_url)?;
            let n_rows = store.ingest_run(experiment_id, &run_dir)?;
            store.mark_completed(experiment_id)?;
            println!(
                "\nExperiment '{}' done → id={experiment_id}  ({n_rows} generation-stat rows)",
                cli.name
            );
        }
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "unknown panic".into()
            };
            let mut store = ExperimentStore::connect(&db_url)?;
            store.mark_failed(experiment_id, &msg)?;
            return Err(format!("pool panicked: {msg}").into());
        }
    }

    Ok(())
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run_experiment(&cli) {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
