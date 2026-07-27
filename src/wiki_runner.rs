use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::checkpoint::{
    load_wiki_population, save_wiki_population, CheckpointConfig, ResumeConfig,
};
use crate::fingerprint::FingerprintConfig;
use crate::genetic_code::{GeneticCodeConfig, GeneticCodeKind};
use crate::population::{GenerationStats, PopConfig};
use crate::runner::{run_population_pool, write_fitness_history};
use crate::wiki::{WikiAutomaton, WikiEnvironment};

#[derive(Clone, Debug)]
pub struct WikiRunnerConfig {
    pub total_populations: usize,
    pub max_parallel: usize,
    pub max_generations: usize,
    pub pop_config: PopConfig,
    pub base_population_seed: u64,
    pub checkpoint: CheckpointConfig,
}

impl Default for WikiRunnerConfig {
    fn default() -> Self {
        Self {
            total_populations: 100,
            max_parallel: 12,
            max_generations: 1_000,
            pop_config: PopConfig {
                size: 100,
                state_bits: 8,
                ticks_per_restart: 1_000,
                restarts_per_gen: 1,
                checkpoint_interval: 1_000,
                mutation_rate: 0.01,
                genetic_code: GeneticCodeConfig {
                    kind: GeneticCodeKind::Tsetlin,
                    tsetlin_clauses: 16,
                },
                fingerprint: Some(FingerprintConfig {
                    bits: 4,
                    tournament_k: 4,
                    mutation_rate: 0.01,
                }),
            },
            base_population_seed: 0,
            checkpoint: CheckpointConfig {
                enabled: true,
                generation_interval: 1_000,
            },
        }
    }
}

pub fn run_pool(
    run_dir: &Path,
    environment: Arc<WikiEnvironment>,
    config: &WikiRunnerConfig,
) -> Vec<Vec<GenerationStats>> {
    println!(
        "  WikiText entries: {}   pop_size: {}   gens: {}   restarts/gen: {}   ticks/restart: {}",
        environment.texts().len(),
        config.pop_config.size,
        config.max_generations,
        config.pop_config.restarts_per_gen,
        config.pop_config.ticks_per_restart,
    );
    run_population_pool::<WikiAutomaton, _>(
        run_dir,
        environment,
        config.total_populations,
        config.max_parallel,
        config.max_generations,
        &config.pop_config,
        config.base_population_seed,
        &config.checkpoint,
        save_wiki_population,
    )
}

pub fn run_resumed_population(
    checkpoint: &Path,
    run_dir: &Path,
    environment: &WikiEnvironment,
    config: &WikiRunnerConfig,
) -> Result<Vec<GenerationStats>, Box<dyn std::error::Error>> {
    std::fs::create_dir_all(run_dir)?;
    let pop_dir = run_dir.join("pop_0");
    std::fs::create_dir_all(&pop_dir)?;
    let mut population = load_wiki_population(
        checkpoint,
        environment,
        &ResumeConfig {
            ticks_per_restart: config.pop_config.ticks_per_restart,
            restarts_per_gen: config.pop_config.restarts_per_gen,
            checkpoint_interval: config.checkpoint.generation_interval,
            mutation_rate: config.pop_config.mutation_rate,
            seed: config.base_population_seed,
        },
    )?;
    if config.max_generations < population.generation {
        return Err(format!(
            "target generation {} is below checkpoint generation {}",
            config.max_generations, population.generation
        )
        .into());
    }
    for automaton in &mut population.automata {
        automaton.reset(environment);
    }

    let source_generation = population.generation;
    let started = Instant::now();
    while population.generation < config.max_generations {
        population.run_generation(environment);
        population.evolve(environment);
        let generation = population.generation;
        if config.checkpoint.enabled
            && config.checkpoint.generation_interval > 0
            && generation % config.checkpoint.generation_interval == 0
        {
            let stem = pop_dir.join(format!("gen_{generation:06}"));
            save_wiki_population(&population, environment, &stem)?;
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

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::checkpoint::save_wiki_population;
    use crate::population::PopulationCore;

    use super::*;

    fn test_directory(label: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("terry-{label}-{}-{nonce}", std::process::id()))
    }

    fn tiny_config() -> WikiRunnerConfig {
        WikiRunnerConfig {
            total_populations: 2,
            max_parallel: 2,
            max_generations: 1,
            pop_config: PopConfig {
                size: 4,
                state_bits: 8,
                ticks_per_restart: 3,
                restarts_per_gen: 1,
                checkpoint_interval: 0,
                mutation_rate: 0.01,
                genetic_code: GeneticCodeConfig {
                    kind: GeneticCodeKind::Tsetlin,
                    tsetlin_clauses: 2,
                },
                fingerprint: None,
            },
            base_population_seed: 0,
            checkpoint: CheckpointConfig {
                enabled: false,
                generation_interval: 0,
            },
        }
    }

    #[test]
    fn tiny_wiki_pool_writes_ordered_histories() {
        let directory = test_directory("wiki-pool");
        let environment = Arc::new(
            WikiEnvironment::new("WikiEnv", vec![b"abc".to_vec(), b"def".to_vec()]).unwrap(),
        );

        let histories = run_pool(&directory, environment, &tiny_config());

        assert_eq!(histories.len(), 2);
        assert!(histories.iter().all(|history| history.len() == 1));
        for population_id in 0..2 {
            let path = directory
                .join(format!("pop_{population_id}"))
                .join("fitness_history.json");
            let document: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            assert_eq!(document["pop_id"], population_id);
            assert_eq!(document["history"][0]["generation"], 1);
        }
        std::fs::remove_dir_all(directory).ok();
    }

    #[test]
    fn resume_uses_target_generation_and_preserves_history() {
        let directory = test_directory("wiki-resume");
        std::fs::create_dir_all(&directory).unwrap();
        let environment =
            WikiEnvironment::new("WikiEnv", vec![b"abc".to_vec(), b"def".to_vec()]).unwrap();
        let mut config = tiny_config();
        config.total_populations = 1;
        let mut source =
            PopulationCore::<WikiAutomaton>::new(&environment, config.pop_config.clone(), 10);
        source.run_generation(&environment);
        source.evolve(&environment);
        let checkpoint = directory.join("source");
        save_wiki_population(&source, &environment, &checkpoint).unwrap();

        config.max_generations = 3;
        config.base_population_seed = 99;
        config.checkpoint = CheckpointConfig {
            enabled: true,
            generation_interval: 3,
        };
        let run_dir = directory.join("run");
        let history = run_resumed_population(&checkpoint, &run_dir, &environment, &config).unwrap();

        assert_eq!(history.len(), 3);
        assert_eq!(history.last().unwrap().generation, 3);
        assert!(run_dir.join("pop_0/gen_000003.toml").exists());
        assert!(run_dir.join("pop_0/gen_000003.npz").exists());
        std::fs::remove_dir_all(directory).ok();
    }
}
