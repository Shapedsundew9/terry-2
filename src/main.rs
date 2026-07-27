/// Maze-runner CLI entry point.
///
/// Run with `--help` to see all options.  Each invocation runs one
/// experiment (one pool of populations) and records results to PostgreSQL.
///
/// Example:
///   maze-runner --name baseline-state4 --state-bits 4
///   maze-runner --name baseline-state3 --state-bits 3 --populations 100
use std::path::PathBuf;

use clap::Parser;
use serde_json::json;

use rust_2::checkpoint::{inspect_checkpoint, CheckpointConfig};
use rust_2::experiment::run_tracked_experiment;
use rust_2::fingerprint::FingerprintConfig;
use rust_2::genetic_code::{GeneticCodeConfig, GeneticCodeKind};
use rust_2::population::PopConfig;
use rust_2::runner::{run_pool, run_resumed_population, RunnerConfig};

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

    /// Maze identity used for checkpoint environment validation.
    #[arg(long)]
    maze_name: Option<String>,

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

    /// Resume one population checkpoint (`.toml`/`.npz` suffix optional).
    #[arg(long)]
    resume: Option<PathBuf>,

    /// Genetic code type: "tsetlin", "dict", or "list".
    #[arg(long, default_value = "tsetlin", value_parser = ["tsetlin", "dict", "list"])]
    code_type: String,

    /// Initial number of clauses per Tsetlin response bit.
    #[arg(long, default_value_t = 4)]
    tsetlin_clauses: usize,

    /// Per-bit mutation rate for crossover.
    #[arg(long, default_value_t = 0.01)]
    mutation_rate: f64,

    /// Enable inherited selection fingerprints.
    #[arg(long, default_value_t = false)]
    fingerprint: bool,

    /// Selection-fingerprint width.
    #[arg(long, default_value_t = 4)]
    fingerprint_bits: u8,

    /// Candidate count for fingerprint mate tournaments.
    #[arg(long, default_value_t = 4)]
    fingerprint_tournament_k: usize,

    /// Mutation probability for an inherited fingerprint.
    #[arg(long, default_value_t = 0.01)]
    fingerprint_mutation_rate: f64,
}

fn run_experiment(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    validate_cli(cli)?;
    let resume_summary = cli.resume.as_deref().map(inspect_checkpoint).transpose()?;
    let maze_name = cli
        .maze_name
        .clone()
        .or_else(|| {
            resume_summary
                .as_ref()
                .map(|summary| summary.environment_name.clone())
        })
        .unwrap_or_else(|| "MazeRunnerMaze".into());
    let effective_populations = if resume_summary.is_some() {
        1
    } else {
        cli.populations
    };
    let effective_parallel = if resume_summary.is_some() {
        1
    } else {
        cli.parallel
    };

    let code_kind = match cli.code_type.as_str() {
        "dict" => GeneticCodeKind::Dict,
        "list" => GeneticCodeKind::List,
        _ => GeneticCodeKind::Tsetlin,
    };

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
        genetic_code: GeneticCodeConfig {
            kind: code_kind,
            tsetlin_clauses: cli.tsetlin_clauses,
        },
        fingerprint: cli.fingerprint.then_some(FingerprintConfig {
            bits: cli.fingerprint_bits,
            tournament_k: cli.fingerprint_tournament_k,
            mutation_rate: cli.fingerprint_mutation_rate,
        }),
    };

    let runner_config = RunnerConfig {
        total_populations: effective_populations,
        max_parallel: effective_parallel,
        max_generations: cli.generations,
        pop_config,
        base_population_seed: cli.pop_seed,
        maze_seed: cli.maze_seed,
        maze_name: maze_name.clone(),
        side_length_bits: cli.side_length_bits,
        checkpoint: checkpoint_cfg,
    };

    let params = json!({
        "total_populations": effective_populations,
        "max_parallel": effective_parallel,
        "max_generations": cli.generations,
        "ticks_per_restart": cli.ticks,
        "restarts_per_gen": cli.restarts,
        "population_size": resume_summary.as_ref().map_or(cli.pop_size, |summary| summary.population_size),
        "side_length_bits": cli.side_length_bits,
        "maze_seed": cli.maze_seed,
        "maze_name": maze_name,
        "population_seed": cli.pop_seed,
        "checkpoint_interval": cli.checkpoint_interval,
        "mutation_rate": cli.mutation_rate,
        "automaton_params": {
            "state_bits": resume_summary.as_ref().map_or(cli.state_bits, |summary| summary.state_bits)
        },
        "code_type": resume_summary.as_ref().map_or_else(
            || cli.code_type.clone(),
            |summary| summary.code_type.clone(),
        ),
        "tsetlin_clauses": cli.tsetlin_clauses,
        "fingerprint_enabled": cli.fingerprint,
        "fingerprint_bits": cli.fingerprint_bits,
        "fingerprint_tournament_k": cli.fingerprint_tournament_k,
        "fingerprint_mutation_rate": cli.fingerprint_mutation_rate,
        "resume_checkpoint": cli.resume.as_ref().map(|path| path.display().to_string()),
        "resume_source_generation": resume_summary.as_ref().map(|summary| summary.generation),
        "resume_fingerprint_enabled": resume_summary.as_ref().map(|summary| summary.fingerprint_enabled),
    });

    run_tracked_experiment(
        &cli.name,
        &cli.description,
        &params,
        &cli.base_dir,
        cli.database_url.as_deref(),
        |run_id, run_dir| {
            if let Some(checkpoint) = &cli.resume {
                println!(
                    "\nMaze Runner (Rust) - '{}'\n  resume {} -> generation {}\n  Run ID: {run_id}  dir: {}",
                    cli.name,
                    checkpoint.display(),
                    cli.generations,
                    run_dir.display(),
                );
                run_resumed_population(checkpoint, run_dir, &runner_config)?;
            } else {
                println!(
                    "\nMaze Runner (Rust) - '{}'\n  {} total x {} parallel x {} gens\n  Run ID: {run_id}  dir: {}",
                    cli.name,
                    cli.populations,
                    cli.parallel,
                    cli.generations,
                    run_dir.display(),
                );
                run_pool(run_dir, &runner_config);
            }
            Ok(())
        },
    )
}

fn validate_cli(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    if !(1..=6).contains(&cli.state_bits) {
        return Err("state-bits must be between 1 and 6".into());
    }
    if cli.pop_size < 2 {
        return Err("pop-size must be at least 2".into());
    }
    if cli.populations == 0 && cli.resume.is_none() {
        return Err("populations must be at least 1".into());
    }
    if cli.parallel == 0 {
        return Err("parallel must be at least 1".into());
    }
    if cli.restarts == 0 {
        return Err("restarts must be at least 1".into());
    }
    if cli.side_length_bits < 4 {
        return Err("side-length-bits must be at least 4".into());
    }
    if !cli.mutation_rate.is_finite() || !(0.0..=1.0).contains(&cli.mutation_rate) {
        return Err("mutation-rate must be between 0 and 1".into());
    }
    if cli.tsetlin_clauses == 0 {
        return Err("tsetlin-clauses must be at least 1".into());
    }
    if cli.fingerprint {
        FingerprintConfig {
            bits: cli.fingerprint_bits,
            tournament_k: cli.fingerprint_tournament_k,
            mutation_rate: cli.fingerprint_mutation_rate,
        }
        .validate()?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_defaults_to_tsetlin() {
        let cli = Cli::try_parse_from(["maze-runner", "--name", "test"]).unwrap();
        assert_eq!(cli.code_type, "tsetlin");
        assert_eq!(cli.tsetlin_clauses, 4);
    }

    #[test]
    fn cli_accepts_model_and_resume_overrides() {
        let cli = Cli::try_parse_from([
            "maze-runner",
            "--name",
            "test",
            "--code-type",
            "dict",
            "--resume",
            "checkpoint",
            "--maze-name",
            "ExampleMaze",
        ])
        .unwrap();
        assert_eq!(cli.code_type, "dict");
        assert_eq!(cli.resume, Some(PathBuf::from("checkpoint")));
        assert_eq!(cli.maze_name.as_deref(), Some("ExampleMaze"));
    }

    #[test]
    fn cli_rejects_unknown_model() {
        assert!(
            Cli::try_parse_from(["maze-runner", "--name", "test", "--code-type", "unknown",])
                .is_err()
        );
    }
}
