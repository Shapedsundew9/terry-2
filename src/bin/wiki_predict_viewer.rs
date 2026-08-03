use std::cmp::Ordering;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use clap::Parser;

use rust_2::checkpoint::{inspect_checkpoint, load_wiki_population, ResumeConfig};
use rust_2::wiki::{
    load_wikitext_environment, WIKITEXT_DATASET_CONFIG, WIKITEXT_DATASET_NAME, WIKITEXT_SPLIT,
};

#[derive(Parser, Debug)]
#[command(
    name = "wiki-predict-viewer",
    about = "Load best Wiki checkpoint automaton and show actual vs predicted text",
    long_about = None,
)]
struct Cli {
    /// Explicit checkpoint path (.toml/.npz or stem). Defaults to latest under --base-dir.
    #[arg(long)]
    checkpoint: Option<PathBuf>,

    /// Root directory for auto-discovery when --checkpoint is omitted.
    #[arg(long, default_value = "runs")]
    base_dir: PathBuf,

    /// Wiki text index to preview.
    #[arg(long, default_value_t = 0)]
    text_index: usize,

    /// Maximum number of output lines.
    #[arg(long, default_value_t = 20)]
    lines: usize,

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
}

fn main() {
    let cli = Cli::parse();
    if let Err(error) = run(&cli) {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

fn run(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint_stem = resolve_checkpoint_stem(cli)?;
    let summary = inspect_checkpoint(&checkpoint_stem)?;
    if summary.environment_class != "ByteEnv"
        || summary.automaton_class != "WikiAutomaton"
        || summary.code_type != "GeneticCodeTsetlin"
    {
        return Err("checkpoint must contain a Tsetlin WikiAutomaton population".into());
    }
    if summary.env_bits % 8 != 0 {
        return Err(format!(
            "checkpoint env_bits {} is not byte-aligned",
            summary.env_bits
        )
        .into());
    }

    let observation_bytes = summary.env_bits / 8;
    let loaded = load_wikitext_environment(
        &cli.dataset_name,
        &cli.dataset_config,
        &cli.dataset_split,
        observation_bytes,
        cli.dataset_path.as_deref(),
        cli.cache_dir.as_deref(),
    )?;
    if summary.environment_name != loaded.environment.name {
        return Err(format!(
            "checkpoint environment {:?} does not match loaded environment {:?}",
            summary.environment_name, loaded.environment.name
        )
        .into());
    }

    let population = load_wiki_population(
        &checkpoint_stem,
        &loaded.environment,
        &ResumeConfig {
            ticks_per_restart: 1,
            restarts_per_gen: 1,
            checkpoint_interval: 0,
            mutation_rate_exponent: 7,
            seed: 0,
        },
    )?;

    let checkpoint_fitness = checkpoint_fitnesses(&population);
    let best_index =
        select_best_index(&checkpoint_fitness).ok_or("checkpoint population is empty")?;
    let best = &population.automata[best_index];
    let selected_population = checkpoint_stem
        .parent()
        .and_then(Path::file_name)
        .and_then(OsStr::to_str)
        .and_then(parse_pop_dir);
    let final_generation_max = checkpoint_fitness
        .iter()
        .copied()
        .reduce(f64::max)
        .unwrap_or(0.0);
    let historical_best_max = population
        .fitness_history
        .iter()
        .map(|stats| stats.max_fitness)
        .reduce(f64::max)
        .unwrap_or(final_generation_max);

    let texts = loaded.environment.texts();
    if cli.text_index >= texts.len() {
        return Err(format!(
            "text-index {} out of range for {} entries",
            cli.text_index,
            texts.len()
        )
        .into());
    }

    let actual_text = &texts[cli.text_index];
    let next_predictions =
        predict_next_bytes(best, &loaded.environment, cli.text_index, actual_text.len())?;
    let aligned_predictions = align_predictions_to_actual(actual_text, &next_predictions);
    let preview_stats = compute_preview_stats(actual_text, &aligned_predictions);
    let lines = split_by_actual_lines(actual_text, &aligned_predictions);
    let max_lines = cli.lines.max(1);
    let shown = lines.len().min(max_lines);

    println!("Wiki prediction preview");
    println!("  checkpoint: {}", checkpoint_stem.display());
    if let Some(pop_id) = selected_population {
        println!("  population: pop_{pop_id}");
    }
    println!("  dataset path: {}", loaded.source_path.display());
    println!("  text index: {}", cli.text_index);
    println!("  text bytes: {}", actual_text.len());
    println!("  best automaton index: {}", best_index);
    println!(
        "  best checkpoint fitness (final gen): {:.6}",
        checkpoint_fitness[best_index]
    );
    println!(
        "  final generation max fitness: {:.6}",
        final_generation_max
    );
    println!("  historical best_max fitness: {:.6}", historical_best_max);
    println!(
        "  preview next-byte match rate: {:.3}% ({}/{})",
        preview_stats.match_rate * 100.0,
        preview_stats.matches,
        preview_stats.compared
    );
    println!(
        "  preview predicted spaces: {:.3}% ({}/{})",
        preview_stats.predicted_space_rate * 100.0,
        preview_stats.predicted_spaces,
        preview_stats.compared
    );
    println!("  lines shown: {} / {}", shown, lines.len());
    println!();

    for (line_no, (actual_line, predicted_line)) in lines.iter().take(shown).enumerate() {
        println!("{:>4} | A | {}", line_no + 1, render_line(actual_line));
        println!("     | P | {}", render_line(predicted_line));
    }
    if shown < lines.len() {
        println!();
        println!(
            "... output truncated at {} lines (use --lines to change)",
            max_lines
        );
    }

    Ok(())
}

fn resolve_checkpoint_stem(cli: &Cli) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(path) = &cli.checkpoint {
        return Ok(normalize_checkpoint_stem(path));
    }

    if let Some(run_dir) = find_latest_run_dir(&cli.base_dir)? {
        if let Some(stem) = find_best_run_checkpoint_stem(&run_dir)? {
            return Ok(stem);
        }
    }

    find_latest_checkpoint_stem(&cli.base_dir)?.ok_or_else(|| {
        format!(
            "no checkpoint files found under {} (expected pop_*/gen_*.toml)",
            cli.base_dir.display()
        )
        .into()
    })
}

fn find_latest_run_dir(base_dir: &Path) -> io::Result<Option<PathBuf>> {
    if !base_dir.exists() {
        return Ok(None);
    }
    let mut best: Option<PathBuf> = None;
    for entry in fs::read_dir(base_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let path = entry.path();
        let Some(name) = path.file_name().and_then(OsStr::to_str) else {
            continue;
        };
        if !is_run_dir_name(name) {
            continue;
        }
        match &best {
            None => best = Some(path),
            Some(current) => {
                if path > *current {
                    best = Some(path);
                }
            }
        }
    }
    Ok(best)
}

fn is_run_dir_name(name: &str) -> bool {
    let Some((timestamp, suffix)) = name.split_once('_') else {
        return false;
    };
    if timestamp.len() != 15 || !timestamp.starts_with("20") {
        return false;
    }
    let (date, time) = timestamp.split_at(8);
    if !date.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    if !time.starts_with('T') || !time[1..].chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    suffix.len() == 6 && suffix.chars().all(|c| c.is_ascii_hexdigit())
}

fn find_best_run_checkpoint_stem(run_dir: &Path) -> io::Result<Option<PathBuf>> {
    struct Candidate {
        pop_id: usize,
        best_max: f64,
        checkpoint_stem: PathBuf,
    }

    let mut best: Option<Candidate> = None;
    for entry in fs::read_dir(run_dir)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let pop_dir = entry.path();
        let Some(pop_name) = pop_dir.file_name().and_then(OsStr::to_str) else {
            continue;
        };
        let Some(pop_id) = parse_pop_dir(pop_name) else {
            continue;
        };
        let Some(checkpoint_stem) = latest_checkpoint_in_pop_dir(&pop_dir)? else {
            continue;
        };
        let best_max = read_best_max_fitness(&pop_dir.join("fitness_history.json"))
            .unwrap_or(f64::NEG_INFINITY);

        let candidate = Candidate {
            pop_id,
            best_max,
            checkpoint_stem,
        };

        match &best {
            None => best = Some(candidate),
            Some(current) => {
                let better = candidate.best_max > current.best_max
                    || (candidate.best_max == current.best_max
                        && candidate.pop_id < current.pop_id);
                if better {
                    best = Some(candidate);
                }
            }
        }
    }

    Ok(best.map(|candidate| candidate.checkpoint_stem))
}

fn parse_pop_dir(name: &str) -> Option<usize> {
    name.strip_prefix("pop_")?.parse::<usize>().ok()
}

fn latest_checkpoint_in_pop_dir(pop_dir: &Path) -> io::Result<Option<PathBuf>> {
    let mut best: Option<PathBuf> = None;
    for entry in fs::read_dir(pop_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !entry.file_type()?.is_file() || !is_checkpoint_toml_path(&path) {
            continue;
        }
        let stem = normalize_checkpoint_stem(&path);
        match &best {
            None => best = Some(stem),
            Some(current) => {
                if stem > *current {
                    best = Some(stem);
                }
            }
        }
    }
    Ok(best)
}

fn read_best_max_fitness(path: &Path) -> Option<f64> {
    let content = fs::read_to_string(path).ok()?;
    let document: serde_json::Value = serde_json::from_str(&content).ok()?;
    let history = document.get("history")?.as_array()?;
    history
        .iter()
        .filter_map(|entry| entry.get("max_fitness").and_then(serde_json::Value::as_f64))
        .reduce(f64::max)
}

fn normalize_checkpoint_stem(path: &Path) -> PathBuf {
    match path.extension().and_then(OsStr::to_str) {
        Some("toml") | Some("npz") => path.with_extension(""),
        _ => path.to_path_buf(),
    }
}

fn find_latest_checkpoint_stem(base_dir: &Path) -> io::Result<Option<PathBuf>> {
    if !base_dir.exists() {
        return Ok(None);
    }
    let mut stack = vec![base_dir.to_path_buf()];
    let mut best: Option<(SystemTime, PathBuf)> = None;

    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                stack.push(path);
                continue;
            }
            if !file_type.is_file() || !is_checkpoint_toml_path(&path) {
                continue;
            }
            let modified = entry
                .metadata()
                .and_then(|metadata| metadata.modified())
                .unwrap_or(UNIX_EPOCH);
            let stem = normalize_checkpoint_stem(&path);
            match &best {
                None => best = Some((modified, stem)),
                Some((current_time, current_path)) => {
                    if modified > *current_time
                        || (modified == *current_time && stem > *current_path)
                    {
                        best = Some((modified, stem));
                    }
                }
            }
        }
    }

    Ok(best.map(|(_, stem)| stem))
}

fn is_checkpoint_toml_path(path: &Path) -> bool {
    let Some(extension) = path.extension().and_then(OsStr::to_str) else {
        return false;
    };
    if extension != "toml" {
        return false;
    }

    let Some(file_name) = path.file_name().and_then(OsStr::to_str) else {
        return false;
    };
    if !file_name.starts_with("gen_") {
        return false;
    }

    let Some(parent_name) = path
        .parent()
        .and_then(Path::file_name)
        .and_then(OsStr::to_str)
    else {
        return false;
    };
    parent_name.starts_with("pop_")
}

fn predict_next_bytes(
    automaton: &rust_2::wiki::WikiAutomaton,
    environment: &rust_2::wiki::WikiEnvironment,
    text_index: usize,
    text_len: usize,
) -> Result<Vec<u8>, String> {
    let tsetlin = automaton
        .genetic_code
        .as_tsetlin()
        .ok_or_else(|| "Wiki checkpoint automaton must use GeneticCodeTsetlin".to_string())?;
    let mut predictions = Vec::with_capacity(text_len);
    let state_bits = automaton.state_bits;
    let state_mask = (1u64 << state_bits) - 1;
    let mut internal_state = automaton.internal_state & state_mask;
    let observation_bits = environment.observation_bits();

    for byte_index in 0..text_len {
        let observation = environment.observation(text_index, byte_index);
        let input_code = (internal_state << observation_bits) | observation;
        let output_code = tsetlin.evaluate(input_code);
        internal_state = output_code & state_mask;
        predictions.push((output_code >> state_bits) as u8);
    }

    Ok(predictions)
}

fn checkpoint_fitnesses(population: &rust_2::wiki::WikiPopulation) -> Vec<f64> {
    population
        .fitness_history
        .last()
        .filter(|stats| stats.fitnesses.len() == population.automata.len())
        .map(|stats| stats.fitnesses.clone())
        .unwrap_or_else(|| {
            population
                .automata
                .iter()
                .map(|automaton| automaton.fitness)
                .collect()
        })
}

fn select_best_index(fitnesses: &[f64]) -> Option<usize> {
    fitnesses
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap_or(Ordering::Equal))
        .map(|(index, _)| index)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PreviewStats {
    matches: usize,
    compared: usize,
    match_rate: f64,
    predicted_spaces: usize,
    predicted_space_rate: f64,
}

fn compute_preview_stats(actual: &[u8], aligned_predictions: &[u8]) -> PreviewStats {
    let compared = actual
        .len()
        .min(aligned_predictions.len())
        .saturating_sub(1);
    if compared == 0 {
        return PreviewStats {
            matches: 0,
            compared: 0,
            match_rate: 0.0,
            predicted_spaces: 0,
            predicted_space_rate: 0.0,
        };
    }

    let actual_slice = &actual[1..=compared];
    let predicted_slice = &aligned_predictions[1..=compared];
    let matches = actual_slice
        .iter()
        .zip(predicted_slice.iter())
        .filter(|(actual_byte, predicted_byte)| actual_byte == predicted_byte)
        .count();
    let predicted_spaces = predicted_slice
        .iter()
        .filter(|&&predicted_byte| predicted_byte == b' ')
        .count();
    let compared_f64 = compared as f64;

    PreviewStats {
        matches,
        compared,
        match_rate: matches as f64 / compared_f64,
        predicted_spaces,
        predicted_space_rate: predicted_spaces as f64 / compared_f64,
    }
}

fn align_predictions_to_actual(actual: &[u8], predictions: &[u8]) -> Vec<u8> {
    if actual.is_empty() {
        return Vec::new();
    }
    let mut aligned = Vec::with_capacity(actual.len());
    aligned.push(b'~');
    for index in 1..actual.len() {
        aligned.push(predictions[index - 1]);
    }
    aligned
}

fn split_by_actual_lines(actual: &[u8], predicted_aligned: &[u8]) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut rows = Vec::new();
    let mut actual_line = Vec::new();
    let mut predicted_line = Vec::new();

    for (actual_byte, predicted_byte) in actual.iter().zip(predicted_aligned.iter()) {
        if *actual_byte == b'\n' {
            rows.push((actual_line, predicted_line));
            actual_line = Vec::new();
            predicted_line = Vec::new();
            continue;
        }
        actual_line.push(*actual_byte);
        predicted_line.push(*predicted_byte);
    }

    if !actual_line.is_empty() || rows.is_empty() {
        rows.push((actual_line, predicted_line));
    }

    rows
}

fn render_line(line: &[u8]) -> String {
    line.iter()
        .flat_map(|byte| std::ascii::escape_default(*byte))
        .map(char::from)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rust_2::genetic_code::{GeneticCodeConfig, GeneticCodeKind};
    use rust_2::population::{GenerationStats, PopConfig};
    use rust_2::wiki::{WikiEnvironment, WikiPopulation};

    use super::*;

    #[test]
    fn normalize_checkpoint_stem_handles_supported_suffixes() {
        assert_eq!(
            normalize_checkpoint_stem(Path::new("runs/id/pop_0/gen_000001.toml")),
            PathBuf::from("runs/id/pop_0/gen_000001")
        );
        assert_eq!(
            normalize_checkpoint_stem(Path::new("runs/id/pop_0/gen_000001.npz")),
            PathBuf::from("runs/id/pop_0/gen_000001")
        );
        assert_eq!(
            normalize_checkpoint_stem(Path::new("runs/id/pop_0/gen_000001")),
            PathBuf::from("runs/id/pop_0/gen_000001")
        );
    }

    #[test]
    fn checkpoint_filter_accepts_only_pop_gen_toml() {
        assert!(is_checkpoint_toml_path(Path::new(
            "runs/r/pop_0/gen_000001.toml"
        )));
        assert!(!is_checkpoint_toml_path(Path::new(
            "runs/r/pop_0/notes.toml"
        )));
        assert!(!is_checkpoint_toml_path(Path::new(
            "runs/r/other/gen_000001.toml"
        )));
        assert!(!is_checkpoint_toml_path(Path::new(
            "runs/r/pop_0/gen_000001.npz"
        )));
    }

    #[test]
    fn latest_checkpoint_prefers_newer_file() {
        let root = test_directory("latest");
        let older = root.join("run_a/pop_0");
        let newer = root.join("run_b/pop_1");
        fs::create_dir_all(&older).unwrap();
        fs::create_dir_all(&newer).unwrap();
        fs::write(older.join("gen_000001.toml"), "").unwrap();
        std::thread::sleep(Duration::from_millis(25));
        fs::write(newer.join("gen_000002.toml"), "").unwrap();

        let found = find_latest_checkpoint_stem(&root).unwrap().unwrap();
        assert_eq!(found, newer.join("gen_000002"));

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn default_discovery_uses_latest_run_best_population() {
        let root = test_directory("default-run-best");
        let latest_run = root.join("20260802T215217_62ce94");
        let older_run = root.join("20260801T120000_aaaaaa");
        fs::create_dir_all(older_run.join("pop_0")).unwrap();
        fs::create_dir_all(latest_run.join("pop_0")).unwrap();
        fs::create_dir_all(latest_run.join("pop_11")).unwrap();

        fs::write(older_run.join("pop_0/gen_000040.toml"), "").unwrap();
        fs::write(latest_run.join("pop_0/gen_000040.toml"), "").unwrap();
        fs::write(latest_run.join("pop_11/gen_000040.toml"), "").unwrap();

        fs::write(
            latest_run.join("pop_0/fitness_history.json"),
            r#"{"history":[{"max_fitness":0.263}]}"#,
        )
        .unwrap();
        fs::write(
            latest_run.join("pop_11/fitness_history.json"),
            r#"{"history":[{"max_fitness":0.250}]}"#,
        )
        .unwrap();

        let cli = Cli {
            checkpoint: None,
            base_dir: root.clone(),
            text_index: 0,
            lines: 20,
            dataset_name: WIKITEXT_DATASET_NAME.to_string(),
            dataset_config: WIKITEXT_DATASET_CONFIG.to_string(),
            dataset_split: WIKITEXT_SPLIT.to_string(),
            dataset_path: None,
            cache_dir: None,
        };

        let resolved = resolve_checkpoint_stem(&cli).unwrap();
        assert_eq!(resolved, latest_run.join("pop_0/gen_000040"));

        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn run_dir_name_validation_matches_expected_shape() {
        assert!(is_run_dir_name("20260802T215217_62ce94"));
        assert!(!is_run_dir_name("20260802_215217_62ce94"));
        assert!(!is_run_dir_name("run_20260802T215217_62ce94"));
        assert!(!is_run_dir_name("20260802T215217_zzzzzz"));
    }

    #[test]
    fn parse_pop_dir_extracts_numeric_suffix() {
        assert_eq!(parse_pop_dir("pop_0"), Some(0));
        assert_eq!(parse_pop_dir("pop_11"), Some(11));
        assert_eq!(parse_pop_dir("pop_x"), None);
        assert_eq!(parse_pop_dir("population_1"), None);
    }

    #[test]
    fn split_and_align_respect_actual_line_boundaries() {
        let actual = b"ab\ncd";
        let predicted = vec![b'x', b'y', b'z', b'w'];
        let aligned = align_predictions_to_actual(actual, &predicted);
        let rows = split_by_actual_lines(actual, &aligned);

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, b"ab");
        assert_eq!(rows[0].1, b"~x");
        assert_eq!(rows[1].0, b"cd");
        assert_eq!(rows[1].1, b"zw");
    }

    #[test]
    fn preview_stats_report_match_and_space_rates() {
        let actual = b" ab";
        let aligned = b"~ a";
        let stats = compute_preview_stats(actual, aligned);

        assert_eq!(stats.compared, 2);
        assert_eq!(stats.matches, 0);
        assert_eq!(stats.predicted_spaces, 1);
        assert!((stats.match_rate - 0.0).abs() < 1e-12);
        assert!((stats.predicted_space_rate - 0.5).abs() < 1e-12);
    }

    #[test]
    fn best_index_prefers_latest_fitness_history() {
        let environment = WikiEnvironment::new("WikiEnv", vec![b"abc".to_vec()]).unwrap();
        let config = PopConfig {
            size: 3,
            state_bits: 8,
            ticks_per_restart: 1,
            restarts_per_gen: 1,
            checkpoint_interval: 0,
            mutation_rate_exponent: 7,
            genetic_code: GeneticCodeConfig {
                kind: GeneticCodeKind::Tsetlin,
                tsetlin_clauses: 2,
            },
            fingerprint: None,
        };
        let mut population = WikiPopulation::new(&environment, config, 42);
        for automaton in &mut population.automata {
            automaton.fitness = 0.0;
        }
        population.fitness_history.push(GenerationStats {
            generation: 1,
            min_fitness: 0.1,
            max_fitness: 0.8,
            mean_fitness: 0.4,
            duration_s: 0.0,
            fitnesses: vec![0.1, 0.8, 0.3],
        });

        let fitnesses = checkpoint_fitnesses(&population);
        assert_eq!(fitnesses, vec![0.1, 0.8, 0.3]);
        assert_eq!(select_best_index(&fitnesses), Some(1));
    }

    #[test]
    fn best_index_falls_back_to_automaton_fitness_without_history() {
        let environment = WikiEnvironment::new("WikiEnv", vec![b"abc".to_vec()]).unwrap();
        let config = PopConfig {
            size: 2,
            state_bits: 8,
            ticks_per_restart: 1,
            restarts_per_gen: 1,
            checkpoint_interval: 0,
            mutation_rate_exponent: 7,
            genetic_code: GeneticCodeConfig {
                kind: GeneticCodeKind::Tsetlin,
                tsetlin_clauses: 2,
            },
            fingerprint: None,
        };
        let mut population = WikiPopulation::new(&environment, config, 7);
        population.automata[0].fitness = 0.25;
        population.automata[1].fitness = 0.75;

        let fitnesses = checkpoint_fitnesses(&population);
        assert_eq!(fitnesses, vec![0.25, 0.75]);
        assert_eq!(select_best_index(&fitnesses), Some(1));
    }

    fn test_directory(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "terry-wiki-predict-{label}-{}-{nonce}",
            std::process::id()
        ))
    }
}
