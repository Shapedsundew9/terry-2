use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use serde_json::json;

use rust_2::checkpoint::{inspect_checkpoint, CheckpointConfig};
use rust_2::experiment::run_tracked_experiment;
use rust_2::fingerprint::FingerprintConfig;
use rust_2::genetic_code::{GeneticCodeConfig, GeneticCodeKind};
use rust_2::population::PopConfig;
use rust_2::wiki::{
    load_wikitext_environment, WIKITEXT_DATASET_CONFIG, WIKITEXT_DATASET_NAME, WIKITEXT_SPLIT,
};
use rust_2::wiki_runner::{run_pool, run_resumed_population, WikiRunnerConfig};

#[derive(Parser, Debug)]
#[command(
    name = "wiki-runner",
    about = "High-performance WikiText byte-prediction evolution runner (Rust)",
    long_about = None,
)]
struct Cli {
    #[arg(long)]
    name: String,

    #[arg(long, default_value = "")]
    description: String,

    #[arg(long, default_value = WIKITEXT_DATASET_NAME)]
    dataset_name: String,

    #[arg(long, default_value = WIKITEXT_DATASET_CONFIG)]
    dataset_config: String,

    #[arg(long, default_value = WIKITEXT_SPLIT)]
    dataset_split: String,

    /// Use a local Parquet split and bypass download/cache resolution.
    #[arg(long)]
    dataset_path: Option<PathBuf>,

    /// Cache root; defaults to XDG_CACHE_HOME or ~/.cache.
    #[arg(long)]
    cache_dir: Option<PathBuf>,

    #[arg(long, default_value_t = 100)]
    populations: usize,

    #[arg(long, default_value_t = 12)]
    parallel: usize,

    #[arg(long, default_value_t = 1_000)]
    generations: usize,

    #[arg(long, default_value_t = 1_000)]
    ticks: usize,

    #[arg(long, default_value_t = 1)]
    restarts: usize,

    #[arg(long, default_value_t = 100)]
    pop_size: usize,

    #[arg(long, default_value_t = 8)]
    state_bits: u8,

    /// Number of trailing raw bytes packed into each observation.
    #[arg(long, default_value_t = 2)]
    observation_bytes: u8,

    #[arg(long, default_value_t = 16)]
    tsetlin_clauses: usize,

    #[arg(long, default_value_t = 0)]
    pop_seed: u64,

    /// Write every N generations; zero writes only at the target generation.
    #[arg(long, default_value_t = 1_000)]
    checkpoint_interval: usize,

    #[arg(long, default_value_t = 0.01)]
    mutation_rate: f64,

    /// Disable inherited selection fingerprints (enabled by default).
    #[arg(long)]
    no_fingerprint: bool,

    #[arg(long, default_value_t = 4)]
    fingerprint_bits: u8,

    #[arg(long, default_value_t = 4)]
    fingerprint_tournament_k: usize,

    #[arg(long, default_value_t = 0.01)]
    fingerprint_mutation_rate: f64,

    #[arg(long, default_value = "runs")]
    base_dir: PathBuf,

    #[arg(long)]
    database_url: Option<String>,

    /// Resume one Wiki population checkpoint (`.toml`/`.npz` suffix optional).
    #[arg(long)]
    resume: Option<PathBuf>,
}

fn run_experiment(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    validate_cli(cli)?;
    let resume_summary = cli.resume.as_deref().map(inspect_checkpoint).transpose()?;
    if let Some(summary) = &resume_summary {
        if summary.environment_class != "ByteEnv"
            || summary.automaton_class != "WikiAutomaton"
            || summary.code_type != "GeneticCodeTsetlin"
        {
            return Err("resume checkpoint must contain a Tsetlin WikiAutomaton population".into());
        }
        if summary.env_bits != cli.observation_bytes * 8 {
            return Err(format!(
                "resume checkpoint uses {} observation bytes; pass --observation-bytes {}",
                summary.env_bits / 8,
                summary.env_bits / 8,
            )
            .into());
        }
    }

    let loaded = load_wikitext_environment(
        &cli.dataset_name,
        &cli.dataset_config,
        &cli.dataset_split,
        cli.observation_bytes,
        cli.dataset_path.as_deref(),
        cli.cache_dir.as_deref(),
    )?;
    if resume_summary
        .as_ref()
        .is_some_and(|summary| summary.environment_name != loaded.environment.name)
    {
        return Err("resume checkpoint environment name does not match WikiEnv".into());
    }

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
    let checkpoint = CheckpointConfig {
        enabled: true,
        generation_interval: if cli.checkpoint_interval == 0 {
            cli.generations
        } else {
            cli.checkpoint_interval
        },
    };
    let fingerprint = (!cli.no_fingerprint).then_some(FingerprintConfig {
        bits: cli.fingerprint_bits,
        tournament_k: cli.fingerprint_tournament_k,
        mutation_rate: cli.fingerprint_mutation_rate,
    });
    let tsetlin_clauses = resume_summary
        .as_ref()
        .and_then(|summary| summary.tsetlin_clauses)
        .unwrap_or(cli.tsetlin_clauses);
    let pop_config = PopConfig {
        size: resume_summary
            .as_ref()
            .map_or(cli.pop_size, |summary| summary.population_size),
        state_bits: resume_summary
            .as_ref()
            .map_or(cli.state_bits, |summary| summary.state_bits),
        ticks_per_restart: cli.ticks,
        restarts_per_gen: cli.restarts,
        checkpoint_interval: cli.checkpoint_interval,
        mutation_rate: cli.mutation_rate,
        genetic_code: GeneticCodeConfig {
            kind: GeneticCodeKind::Tsetlin,
            tsetlin_clauses,
        },
        fingerprint,
    };
    let runner_config = WikiRunnerConfig {
        total_populations: effective_populations,
        max_parallel: effective_parallel,
        max_generations: cli.generations,
        pop_config,
        base_population_seed: cli.pop_seed,
        checkpoint,
    };
    let source_path = loaded.source_path.display().to_string();
    let params = json!({
        "total_populations": effective_populations,
        "max_parallel": effective_parallel,
        "max_generations": cli.generations,
        "ticks_per_restart": cli.ticks,
        "restarts_per_gen": cli.restarts,
        "population_size": runner_config.pop_config.size,
        "population_seed": cli.pop_seed,
        "checkpoint_interval": cli.checkpoint_interval,
        "mutation_rate": cli.mutation_rate,
        "dataset_name": cli.dataset_name,
        "dataset_config": cli.dataset_config,
        "dataset_split": cli.dataset_split,
        "dataset_path": cli.dataset_path.as_ref().map(|path| path.display().to_string()),
        "resolved_dataset_path": source_path,
        "automaton_params": {
            "observation_bytes": cli.observation_bytes,
            "env_bits": cli.observation_bytes * 8,
            "state_bits": runner_config.pop_config.state_bits,
            "resp_bits": 8,
            "num_clauses": tsetlin_clauses,
        },
        "code_type": "tsetlin",
        "fingerprint_enabled": !cli.no_fingerprint,
        "fingerprint_bits": cli.fingerprint_bits,
        "fingerprint_tournament_k": cli.fingerprint_tournament_k,
        "fingerprint_mutation_rate": cli.fingerprint_mutation_rate,
        "resume_checkpoint": cli.resume.as_ref().map(|path| path.display().to_string()),
        "resume_source_generation": resume_summary.as_ref().map(|summary| summary.generation),
        "resume_fingerprint_enabled": resume_summary.as_ref().map(|summary| summary.fingerprint_enabled),
    });
    let environment = Arc::new(loaded.environment);

    run_tracked_experiment(
        &cli.name,
        &cli.description,
        &params,
        &cli.base_dir,
        cli.database_url.as_deref(),
        |run_id, run_dir| {
            if let Some(checkpoint) = &cli.resume {
                println!(
                    "\nWikiText Runner (Rust) - '{}'\n  resume {} -> generation {}\n  Dataset: {}\n  Run ID: {run_id}  dir: {}",
                    cli.name,
                    checkpoint.display(),
                    cli.generations,
                    source_path,
                    run_dir.display(),
                );
                run_resumed_population(checkpoint, run_dir, environment.as_ref(), &runner_config)?;
            } else {
                println!(
                    "\nWikiText Runner (Rust) - '{}'\n  {} total x {} parallel x {} gens\n  Dataset: {}\n  Run ID: {run_id}  dir: {}",
                    cli.name,
                    cli.populations,
                    cli.parallel,
                    cli.generations,
                    source_path,
                    run_dir.display(),
                );
                run_pool(run_dir, Arc::clone(&environment), &runner_config);
            }
            Ok(())
        },
    )
}

fn validate_cli(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    if !(1..=7).contains(&cli.observation_bytes) {
        return Err("observation-bytes must be between 1 and 7".into());
    }
    if cli.state_bits == 0 {
        return Err("state-bits must be at least 1".into());
    }
    let state_bits = u16::from(cli.state_bits);
    let observation_bits = u16::from(cli.observation_bytes) * 8;
    if state_bits + 8 >= 64 {
        return Err("state-bits + 8 prediction bits must be less than 64".into());
    }
    if state_bits + observation_bits >= 64 {
        return Err("state-bits + observation bits must be less than 64".into());
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
    if cli.generations == 0 {
        return Err("generations must be at least 1".into());
    }
    if cli.ticks == 0 {
        return Err("ticks must be at least 1".into());
    }
    if cli.restarts == 0 {
        return Err("restarts must be at least 1".into());
    }
    if cli.tsetlin_clauses == 0 {
        return Err("tsetlin-clauses must be at least 1".into());
    }
    if !cli.mutation_rate.is_finite() || !(0.0..=1.0).contains(&cli.mutation_rate) {
        return Err("mutation-rate must be between 0 and 1".into());
    }
    if !cli.no_fingerprint {
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
    if let Err(error) = run_experiment(&cli) {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_defaults_match_python_wiki_runner() {
        let cli = Cli::try_parse_from(["wiki-runner", "--name", "test"]).unwrap();

        assert_eq!(cli.populations, 100);
        assert_eq!(cli.parallel, 12);
        assert_eq!(cli.generations, 1_000);
        assert_eq!(cli.ticks, 1_000);
        assert_eq!(cli.restarts, 1);
        assert_eq!(cli.pop_size, 100);
        assert_eq!(cli.state_bits, 8);
        assert_eq!(cli.observation_bytes, 2);
        assert_eq!(cli.tsetlin_clauses, 16);
        assert!(!cli.no_fingerprint);
        assert_eq!(cli.dataset_name, WIKITEXT_DATASET_NAME);
        assert_eq!(cli.dataset_config, WIKITEXT_DATASET_CONFIG);
        assert_eq!(cli.dataset_split, WIKITEXT_SPLIT);
    }

    #[test]
    fn cli_accepts_local_dataset_resume_and_fingerprint_override() {
        let cli = Cli::try_parse_from([
            "wiki-runner",
            "--name",
            "test",
            "--dataset-path",
            "train.parquet",
            "--resume",
            "checkpoint",
            "--no-fingerprint",
        ])
        .unwrap();

        assert_eq!(cli.dataset_path, Some(PathBuf::from("train.parquet")));
        assert_eq!(cli.resume, Some(PathBuf::from("checkpoint")));
        assert!(cli.no_fingerprint);
    }

    #[test]
    fn cli_accepts_custom_wiki_widths() {
        let cli = Cli::try_parse_from([
            "wiki-runner",
            "--name",
            "test",
            "--observation-bytes",
            "4",
            "--state-bits",
            "16",
            "--tsetlin-clauses",
            "32",
        ])
        .unwrap();

        validate_cli(&cli).unwrap();
        assert_eq!(cli.observation_bytes, 4);
        assert_eq!(cli.state_bits, 16);
        assert_eq!(cli.tsetlin_clauses, 32);
    }

    #[test]
    fn cli_rejects_widths_that_reach_sixty_four_bits() {
        let output_too_wide = Cli::try_parse_from([
            "wiki-runner",
            "--name",
            "test",
            "--observation-bytes",
            "1",
            "--state-bits",
            "56",
        ])
        .unwrap();
        let input_too_wide = Cli::try_parse_from([
            "wiki-runner",
            "--name",
            "test",
            "--observation-bytes",
            "2",
            "--state-bits",
            "48",
        ])
        .unwrap();

        assert!(validate_cli(&output_too_wide).is_err());
        assert!(validate_cli(&input_too_wide).is_err());
    }
}
