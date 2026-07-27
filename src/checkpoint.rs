/// Checkpoint persistence — TOML + NPZ (NumPy-compatible).
///
/// Writes checkpoints in the same format as Python's `Checkpointable.save`
/// so that `Population.load()` in Python can read Rust-generated checkpoints.
///
/// TOML structure mirrors Python's `Population.to_dict()`.
/// NPZ structure mirrors Python's `Population.to_arrays()`:
///   - `automaton_{i}_keys`       — int64 array
///   - `automaton_{i}_values`     — int64 array
///   - `automaton_{i}_energy_grid`— uint8 array
///   - `fitness_history_fitnesses`— float64 2-D array
use std::fs::File;
use std::io::{self, BufReader, Write};
use std::path::Path;

use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use toml::{Table, Value};

use crate::automaton::MazeAutomaton;
use crate::fingerprint::{FingerprintConfig, SelectionFingerprint};
use crate::genetic_code::{GeneticCode, GeneticCodeConfig, GeneticCodeKind, GeneticCodeTsetlin};
use crate::maze::Maze;
use crate::population::{GenerationStats, PopConfig, Population};
use crate::wiki::{WikiAutomaton, WikiEnvironment, WikiPopulation};

/// Checkpoint configuration.
#[derive(Clone, Debug, Default)]
pub struct CheckpointConfig {
    pub enabled: bool,
    pub generation_interval: usize,
}

impl CheckpointConfig {
    pub fn to_toml_table(&self, base_dir: &str) -> Table {
        let mut t = Table::new();
        t.insert("enabled".into(), Value::Boolean(self.enabled));
        t.insert("base_dir".into(), Value::String(base_dir.into()));
        t.insert(
            "generation_interval".into(),
            Value::Integer(self.generation_interval as i64),
        );
        t
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Write `<stem>.toml` and `<stem>.npz` for `population`.
///
/// Panics if either file cannot be written.
pub fn save_population(population: &Population, maze: &Maze, stem: &Path) -> io::Result<()> {
    let toml_path = stem.with_extension("toml");
    let npz_path = stem.with_extension("npz");

    let toml_doc = build_toml(population, maze);
    let toml_str = toml::to_string(&toml_doc).map_err(|e| io::Error::other(e.to_string()))?;
    std::fs::write(&toml_path, toml_str.as_bytes())?;

    let npz_bytes = build_npz(population)?;
    std::fs::write(&npz_path, &npz_bytes)?;

    Ok(())
}

pub fn save_wiki_population(
    population: &WikiPopulation,
    environment: &WikiEnvironment,
    stem: &Path,
) -> io::Result<()> {
    let toml_path = stem.with_extension("toml");
    let npz_path = stem.with_extension("npz");

    let toml_doc = build_wiki_toml(population, environment);
    let toml_str = toml::to_string(&toml_doc).map_err(|e| io::Error::other(e.to_string()))?;
    std::fs::write(&toml_path, toml_str.as_bytes())?;
    std::fs::write(&npz_path, build_wiki_npz(population)?)?;
    Ok(())
}

#[derive(Clone, Debug)]
pub struct ResumeConfig {
    pub ticks_per_restart: usize,
    pub restarts_per_gen: usize,
    pub checkpoint_interval: usize,
    pub mutation_rate: f64,
    pub seed: u64,
}

#[derive(Clone, Debug)]
pub struct CheckpointSummary {
    pub generation: usize,
    pub population_size: usize,
    pub state_bits: u8,
    pub code_type: String,
    pub tsetlin_clauses: Option<usize>,
    pub environment_class: String,
    pub environment_name: String,
    pub automaton_class: String,
    pub fingerprint_enabled: bool,
}

pub fn inspect_checkpoint(stem: &Path) -> Result<CheckpointSummary, Box<dyn std::error::Error>> {
    let stem = stem.with_extension("");
    let document: Value = toml::from_str(&std::fs::read_to_string(stem.with_extension("toml"))?)?;
    let root = value_table(&document, "checkpoint root")?;
    let meta = child_table(root, "meta")?;
    let environment = child_table(root, "environment")?;
    let automata = root
        .get("automata")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_data("checkpoint is missing automata"))?;
    let first = automata
        .first()
        .ok_or_else(|| invalid_data("checkpoint population is empty"))?;
    let first = value_table(first, "automaton")?;
    let genetic_code = child_table(first, "genetic_code")?;
    Ok(CheckpointSummary {
        generation: usize_value(meta, "generation")?,
        population_size: automata.len(),
        state_bits: u8_value(first, "state_bits")?,
        code_type: string_value(genetic_code, "type")?.to_string(),
        tsetlin_clauses: genetic_code
            .get("num_clauses")
            .map(|_| usize_value(genetic_code, "num_clauses"))
            .transpose()?,
        environment_class: string_value(environment, "class")?.to_string(),
        environment_name: string_value(environment, "name")?.to_string(),
        automaton_class: string_value(meta, "automaton_class")?.to_string(),
        fingerprint_enabled: root.contains_key("fingerprint_config"),
    })
}

/// Load a schema-v1 Python or Rust population checkpoint.
///
/// The supplied maze is the external environment contract used by Python's
/// loader too. Runtime scheduling and continuation RNG state come from
/// `resume`; model structure and fingerprints come from the checkpoint.
pub fn load_population(
    stem: &Path,
    maze: &Maze,
    resume: &ResumeConfig,
) -> Result<Population, Box<dyn std::error::Error>> {
    if !resume.mutation_rate.is_finite() || !(0.0..=1.0).contains(&resume.mutation_rate) {
        return Err(invalid_data("mutation_rate must be between 0 and 1").into());
    }
    let stem = stem.with_extension("");
    let document: Value = toml::from_str(&std::fs::read_to_string(stem.with_extension("toml"))?)?;
    let root = value_table(&document, "checkpoint root")?;
    let meta = child_table(root, "meta")?;
    if string_value(meta, "class")? != "Population" {
        return Err(invalid_data("checkpoint class must be Population").into());
    }
    let schema_version = integer_value(meta, "schema_version")?;
    if schema_version != 1 {
        return Err(invalid_data(format!(
            "unsupported checkpoint schema version {schema_version}"
        ))
        .into());
    }
    let environment = child_table(root, "environment")?;
    if string_value(environment, "class")? != "Maze"
        || string_value(environment, "name")? != maze.name
    {
        return Err(invalid_data(format!(
            "environment mismatch: checkpoint has {}/{:?}, supplied Maze/{:?}",
            string_value(environment, "class")?,
            string_value(environment, "name")?,
            maze.name
        ))
        .into());
    }

    let generation = usize_value(meta, "generation")?;
    let tick_count = u64_value(meta, "tick_count")?;
    let fingerprint_config = root
        .get("fingerprint_config")
        .map(
            |value| -> Result<FingerprintConfig, Box<dyn std::error::Error>> {
                let table = value_table(value, "fingerprint_config")?;
                let config = FingerprintConfig {
                    bits: u8_value_default(table, "bits", 32)?,
                    tournament_k: usize_value_default(table, "tournament_k", 1)?,
                    mutation_rate: float_value_default(table, "mutation_rate", 0.01)?,
                };
                config.validate().map_err(invalid_data)?;
                Ok(config)
            },
        )
        .transpose()?;

    let file = File::open(stem.with_extension("npz"))?;
    let mut npz = NpzReader::new(BufReader::new(file))?;
    let history_fitnesses: Option<Array2<f64>> = npz.by_name("fitness_history_fitnesses.npy").ok();
    let history_meta = root
        .get("fitness_history")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if let Some(fitnesses) = &history_fitnesses {
        if fitnesses.nrows() != history_meta.len() {
            return Err(invalid_data("fitness history row count does not match TOML").into());
        }
    }
    let mut fitness_history = Vec::with_capacity(history_meta.len());
    for (index, value) in history_meta.iter().enumerate() {
        let table = value_table(value, "fitness_history entry")?;
        fitness_history.push(GenerationStats {
            generation: usize_value(table, "generation")?,
            min_fitness: float_value(table, "min_fitness")?,
            max_fitness: float_value(table, "max_fitness")?,
            mean_fitness: float_value(table, "mean_fitness")?,
            duration_s: float_value_default(table, "duration_s", 0.0)?,
            fitnesses: history_fitnesses
                .as_ref()
                .map(|values| values.row(index).to_vec())
                .unwrap_or_default(),
        });
    }

    let automata_meta = root
        .get("automata")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_data("checkpoint is missing automata"))?;
    if automata_meta.is_empty() {
        return Err(invalid_data("checkpoint population is empty").into());
    }
    if let Some(fitnesses) = &history_fitnesses {
        if fitnesses.ncols() != automata_meta.len() {
            return Err(invalid_data("fitness history width does not match population").into());
        }
    }

    let mut automata = Vec::with_capacity(automata_meta.len());
    let mut state_bits = None;
    let mut code_kind = None;
    let mut initial_clauses = 10usize;
    for (index, value) in automata_meta.iter().enumerate() {
        let table = value_table(value, "automaton")?;
        let current_state_bits = u8_value(table, "state_bits")?;
        if state_bits
            .replace(current_state_bits)
            .is_some_and(|bits| bits != current_state_bits)
        {
            return Err(invalid_data("checkpoint automata have mixed state widths").into());
        }
        if u8_value_default(table, "env_bits", 9)? != 9
            || u8_value_default(table, "resp_bits", 2)? != 2
        {
            return Err(
                invalid_data("Rust MazeAutomaton requires env_bits=9 and resp_bits=2").into(),
            );
        }
        let genetic_meta = child_table(table, "genetic_code")?;
        let metadata_output_bits = genetic_meta
            .get("resp_bits")
            .map(|_| u8_value(genetic_meta, "resp_bits"))
            .transpose()?;
        let seed = optional_seed(genetic_meta);
        let prefix = format!("automaton_{index}_");
        let genetic_code = match string_value(genetic_meta, "type")? {
            "GeneticCodeDict" => {
                set_common_kind(&mut code_kind, GeneticCodeKind::Dict)?;
                let output_bits = metadata_output_bits.unwrap_or(1);
                let keys: Array1<i64> = npz.by_name(&format!("{prefix}keys.npy"))?;
                let values: Array1<i64> = npz.by_name(&format!("{prefix}values.npy"))?;
                if keys.len() != values.len() {
                    return Err(invalid_data("Dict checkpoint key/value lengths differ").into());
                }
                let entries = keys
                    .iter()
                    .zip(values.iter())
                    .map(|(&key, &value)| {
                        let key = u32::try_from(key)
                            .map_err(|_| invalid_data("Dict checkpoint key is outside u32"))?;
                        let value = u16::try_from(value)
                            .map_err(|_| invalid_data("Dict checkpoint value is outside u16"))?;
                        Ok((key, value))
                    })
                    .collect::<Result<Vec<_>, io::Error>>()?;
                GeneticCode::from_dict_entries(entries, output_bits, seed)
            }
            "GeneticCodeList" => {
                set_common_kind(&mut code_kind, GeneticCodeKind::List)?;
                let output_bits = metadata_output_bits.unwrap_or(1);
                let values: Array1<i64> = npz.by_name(&format!("{prefix}values.npy"))?;
                let values = values
                    .iter()
                    .map(|&value| {
                        u16::try_from(value)
                            .map_err(|_| invalid_data("List checkpoint value is outside u16"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                GeneticCode::from_list_values(values, output_bits, seed)
            }
            "GeneticCodeTsetlin" => {
                set_common_kind(&mut code_kind, GeneticCodeKind::Tsetlin)?;
                let positive: Array2<u64> = npz.by_name(&format!("{prefix}w_pos.npy"))?;
                let negative: Array2<u64> = npz.by_name(&format!("{prefix}w_neg.npy"))?;
                if positive.raw_dim() != negative.raw_dim() {
                    return Err(
                        invalid_data("Tsetlin positive/negative matrix shapes differ").into(),
                    );
                }
                let output_bits = metadata_output_bits.unwrap_or(u8::try_from(positive.nrows())?);
                if positive.nrows() != output_bits as usize {
                    return Err(invalid_data("Tsetlin matrix rows do not match resp_bits").into());
                }
                let clauses = positive.ncols();
                let metadata_clauses = usize_value_default(genetic_meta, "num_clauses", clauses)?;
                if genetic_meta.contains_key("num_clauses") && metadata_clauses != clauses {
                    return Err(
                        invalid_data("Tsetlin matrix columns do not match num_clauses").into(),
                    );
                }
                let input_bits = u8_value_default(genetic_meta, "input_bits", 64)?;
                if index == 0 {
                    initial_clauses = clauses;
                }
                GeneticCode::Tsetlin(GeneticCodeTsetlin::from_masks(
                    positive.iter().copied().collect(),
                    negative.iter().copied().collect(),
                    output_bits,
                    clauses,
                    input_bits,
                    seed,
                )?)
            }
            other => {
                return Err(invalid_data(format!("unknown genetic-code type {other:?}")).into())
            }
        };

        let coords = table
            .get("coords")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_data("automaton coords must be an array"))?;
        if coords.len() != 3 {
            return Err(invalid_data("MazeAutomaton coords must contain x, y, orientation").into());
        }
        let coord = |position: usize| -> Result<usize, io::Error> {
            coords[position]
                .as_integer()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| invalid_data("automaton coordinate is invalid"))
        };
        let energy_grid: Array1<u8> = npz.by_name(&format!("{prefix}energy_grid.npy"))?;
        let fingerprint = match (
            table.get("fingerprint_bits").and_then(Value::as_integer),
            table.get("fingerprint_value").and_then(Value::as_integer),
        ) {
            (Some(bits), Some(value)) => Some(SelectionFingerprint::with_value(
                u8::try_from(bits).map_err(|_| invalid_data("fingerprint bits are invalid"))?,
                u64::try_from(value).map_err(|_| invalid_data("fingerprint value is invalid"))?,
            )),
            (None, None) => None,
            _ => return Err(invalid_data("fingerprint bits/value must appear together").into()),
        };
        let mut automaton = MazeAutomaton::restore(
            genetic_code,
            maze,
            current_state_bits,
            resume.seed.wrapping_add(index as u64 + 1),
            coord(0)?,
            coord(1)?,
            u8::try_from(coord(2)?).map_err(|_| invalid_data("orientation is invalid"))?,
            u8_value_default(table, "internal_state", 0)?,
            i32_value(table, "energy")?,
            energy_grid.to_vec(),
            float_value(table, "fitness")?,
            i32_value_default(table, "last_action", -1)?,
            fingerprint,
        )?;
        automaton.id = index as u64;
        automata.push(automaton);
    }

    let state_bits = state_bits.expect("non-empty automata checked above");
    let kind = code_kind.expect("non-empty automata checked above");
    let config = PopConfig {
        size: automata.len(),
        state_bits,
        ticks_per_restart: resume.ticks_per_restart,
        restarts_per_gen: resume.restarts_per_gen,
        checkpoint_interval: resume.checkpoint_interval,
        mutation_rate: resume.mutation_rate,
        genetic_code: GeneticCodeConfig {
            kind,
            tsetlin_clauses: initial_clauses,
        },
        fingerprint: fingerprint_config,
    };
    Ok(Population::restore(
        automata,
        generation,
        tick_count,
        fitness_history,
        config,
        resume.seed,
    ))
}

pub fn load_wiki_population(
    stem: &Path,
    environment: &WikiEnvironment,
    resume: &ResumeConfig,
) -> Result<WikiPopulation, Box<dyn std::error::Error>> {
    if !resume.mutation_rate.is_finite() || !(0.0..=1.0).contains(&resume.mutation_rate) {
        return Err(invalid_data("mutation_rate must be between 0 and 1").into());
    }
    let stem = stem.with_extension("");
    let document: Value = toml::from_str(&std::fs::read_to_string(stem.with_extension("toml"))?)?;
    let root = value_table(&document, "checkpoint root")?;
    let meta = child_table(root, "meta")?;
    if string_value(meta, "class")? != "Population"
        || integer_value(meta, "schema_version")? != 1
        || string_value(meta, "automaton_class")? != "WikiAutomaton"
    {
        return Err(invalid_data("checkpoint is not a schema-v1 Wiki population").into());
    }
    let environment_meta = child_table(root, "environment")?;
    if string_value(environment_meta, "class")? != "ByteEnv"
        || string_value(environment_meta, "name")? != environment.name
    {
        return Err(invalid_data(format!(
            "environment mismatch: checkpoint has {}/{:?}, supplied ByteEnv/{:?}",
            string_value(environment_meta, "class")?,
            string_value(environment_meta, "name")?,
            environment.name
        ))
        .into());
    }

    let generation = usize_value(meta, "generation")?;
    let tick_count = u64_value(meta, "tick_count")?;
    let fingerprint_config = root
        .get("fingerprint_config")
        .map(
            |value| -> Result<FingerprintConfig, Box<dyn std::error::Error>> {
                let table = value_table(value, "fingerprint_config")?;
                let config = FingerprintConfig {
                    bits: u8_value_default(table, "bits", 32)?,
                    tournament_k: usize_value_default(table, "tournament_k", 1)?,
                    mutation_rate: float_value_default(table, "mutation_rate", 0.01)?,
                };
                config.validate().map_err(invalid_data)?;
                Ok(config)
            },
        )
        .transpose()?;

    let file = File::open(stem.with_extension("npz"))?;
    let mut npz = NpzReader::new(BufReader::new(file))?;
    let history_fitnesses: Option<Array2<f64>> = npz.by_name("fitness_history_fitnesses.npy").ok();
    let history_meta = root
        .get("fitness_history")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if history_fitnesses
        .as_ref()
        .is_some_and(|fitnesses| fitnesses.nrows() != history_meta.len())
    {
        return Err(invalid_data("fitness history row count does not match TOML").into());
    }
    let mut fitness_history = Vec::with_capacity(history_meta.len());
    for (index, value) in history_meta.iter().enumerate() {
        let table = value_table(value, "fitness_history entry")?;
        fitness_history.push(GenerationStats {
            generation: usize_value(table, "generation")?,
            min_fitness: float_value(table, "min_fitness")?,
            max_fitness: float_value(table, "max_fitness")?,
            mean_fitness: float_value(table, "mean_fitness")?,
            duration_s: float_value_default(table, "duration_s", 0.0)?,
            fitnesses: history_fitnesses
                .as_ref()
                .map(|values| values.row(index).to_vec())
                .unwrap_or_default(),
        });
    }

    let automata_meta = root
        .get("automata")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_data("checkpoint is missing automata"))?;
    if automata_meta.is_empty() {
        return Err(invalid_data("checkpoint population is empty").into());
    }
    if history_fitnesses
        .as_ref()
        .is_some_and(|fitnesses| fitnesses.ncols() != automata_meta.len())
    {
        return Err(invalid_data("fitness history width does not match population").into());
    }

    let mut automata = Vec::with_capacity(automata_meta.len());
    let mut state_bits = None;
    let mut num_clauses = None;
    for (index, value) in automata_meta.iter().enumerate() {
        let table = value_table(value, "automaton")?;
        if u8_value_default(table, "env_bits", 16)? != 16
            || u8_value_default(table, "resp_bits", 8)? != 8
        {
            return Err(
                invalid_data("Rust WikiAutomaton requires env_bits=16 and resp_bits=8").into(),
            );
        }
        let current_state_bits = u8_value(table, "state_bits")?;
        if state_bits
            .replace(current_state_bits)
            .is_some_and(|bits| bits != current_state_bits)
        {
            return Err(invalid_data("checkpoint automata have mixed state widths").into());
        }

        let genetic_meta = child_table(table, "genetic_code")?;
        if string_value(genetic_meta, "type")? != "GeneticCodeTsetlin" {
            return Err(invalid_data("Wiki checkpoints require GeneticCodeTsetlin").into());
        }
        let prefix = format!("automaton_{index}_");
        let positive: Array2<u64> = npz.by_name(&format!("{prefix}w_pos.npy"))?;
        let negative: Array2<u64> = npz.by_name(&format!("{prefix}w_neg.npy"))?;
        if positive.raw_dim() != negative.raw_dim() {
            return Err(invalid_data("Tsetlin positive/negative matrix shapes differ").into());
        }
        let output_bits =
            u8_value_default(genetic_meta, "resp_bits", u8::try_from(positive.nrows())?)?;
        if output_bits != current_state_bits + 8 || positive.nrows() != output_bits as usize {
            return Err(invalid_data("Tsetlin output width does not match WikiAutomaton").into());
        }
        let clauses = positive.ncols();
        if usize_value_default(genetic_meta, "num_clauses", clauses)? != clauses {
            return Err(invalid_data("Tsetlin matrix columns do not match num_clauses").into());
        }
        if num_clauses
            .replace(clauses)
            .is_some_and(|value| value != clauses)
        {
            return Err(invalid_data("checkpoint automata have mixed clause counts").into());
        }
        let input_bits = u8_value_default(genetic_meta, "input_bits", 24)?;
        if input_bits != current_state_bits + 16 {
            return Err(invalid_data("Tsetlin input width does not match WikiAutomaton").into());
        }
        let genetic_code = GeneticCode::Tsetlin(GeneticCodeTsetlin::from_masks(
            positive.iter().copied().collect(),
            negative.iter().copied().collect(),
            output_bits,
            clauses,
            input_bits,
            optional_seed(genetic_meta),
        )?);

        let coords = table
            .get("coords")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_data("automaton coords must be an array"))?;
        if coords.len() != 2 {
            return Err(
                invalid_data("WikiAutomaton coords must contain text and byte indices").into(),
            );
        }
        let coord = |position: usize| -> Result<usize, io::Error> {
            coords[position]
                .as_integer()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| invalid_data("automaton coordinate is invalid"))
        };
        let fingerprint = match (
            table.get("fingerprint_bits").and_then(Value::as_integer),
            table.get("fingerprint_value").and_then(Value::as_integer),
        ) {
            (Some(bits), Some(value)) => Some(SelectionFingerprint::with_value(
                u8::try_from(bits).map_err(|_| invalid_data("fingerprint bits are invalid"))?,
                u64::try_from(value).map_err(|_| invalid_data("fingerprint value is invalid"))?,
            )),
            (None, None) => None,
            _ => return Err(invalid_data("fingerprint bits/value must appear together").into()),
        };
        let mut automaton = WikiAutomaton::restore(
            genetic_code,
            environment,
            current_state_bits,
            resume.seed.wrapping_add(index as u64 + 1),
            coord(0)?,
            coord(1)?,
            u16_value_default(table, "internal_state", 0)?,
            float_value(table, "fitness")?,
            i32_value_default(table, "last_action", -1)?,
            fingerprint,
        )?;
        automaton.id = index as u64;
        automata.push(automaton);
    }

    let config = PopConfig {
        size: automata.len(),
        state_bits: state_bits.expect("non-empty automata checked above"),
        ticks_per_restart: resume.ticks_per_restart,
        restarts_per_gen: resume.restarts_per_gen,
        checkpoint_interval: resume.checkpoint_interval,
        mutation_rate: resume.mutation_rate,
        genetic_code: GeneticCodeConfig {
            kind: GeneticCodeKind::Tsetlin,
            tsetlin_clauses: num_clauses.expect("non-empty automata checked above"),
        },
        fingerprint: fingerprint_config,
    };
    Ok(WikiPopulation::restore(
        automata,
        generation,
        tick_count,
        fitness_history,
        config,
        resume.seed,
    ))
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn value_table<'a>(value: &'a Value, context: &str) -> Result<&'a Table, io::Error> {
    value
        .as_table()
        .ok_or_else(|| invalid_data(format!("{context} must be a TOML table")))
}

fn child_table<'a>(table: &'a Table, key: &str) -> Result<&'a Table, io::Error> {
    table
        .get(key)
        .ok_or_else(|| invalid_data(format!("checkpoint is missing {key}")))
        .and_then(|value| value_table(value, key))
}

fn string_value<'a>(table: &'a Table, key: &str) -> Result<&'a str, io::Error> {
    table
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_data(format!("{key} must be a string")))
}

fn integer_value(table: &Table, key: &str) -> Result<i64, io::Error> {
    table
        .get(key)
        .and_then(Value::as_integer)
        .ok_or_else(|| invalid_data(format!("{key} must be an integer")))
}

fn usize_value(table: &Table, key: &str) -> Result<usize, io::Error> {
    usize::try_from(integer_value(table, key)?)
        .map_err(|_| invalid_data(format!("{key} must be a nonnegative usize")))
}

fn usize_value_default(table: &Table, key: &str, default: usize) -> Result<usize, io::Error> {
    match table.get(key) {
        Some(_) => usize_value(table, key),
        None => Ok(default),
    }
}

fn u64_value(table: &Table, key: &str) -> Result<u64, io::Error> {
    u64::try_from(integer_value(table, key)?)
        .map_err(|_| invalid_data(format!("{key} must be a nonnegative u64")))
}

fn u8_value(table: &Table, key: &str) -> Result<u8, io::Error> {
    u8::try_from(integer_value(table, key)?)
        .map_err(|_| invalid_data(format!("{key} must fit in u8")))
}

fn u8_value_default(table: &Table, key: &str, default: u8) -> Result<u8, io::Error> {
    match table.get(key) {
        Some(_) => u8_value(table, key),
        None => Ok(default),
    }
}

fn u16_value_default(table: &Table, key: &str, default: u16) -> Result<u16, io::Error> {
    match table.get(key) {
        Some(_) => u16::try_from(integer_value(table, key)?)
            .map_err(|_| invalid_data(format!("{key} must fit in u16"))),
        None => Ok(default),
    }
}

fn i32_value(table: &Table, key: &str) -> Result<i32, io::Error> {
    i32::try_from(integer_value(table, key)?)
        .map_err(|_| invalid_data(format!("{key} must fit in i32")))
}

fn i32_value_default(table: &Table, key: &str, default: i32) -> Result<i32, io::Error> {
    match table.get(key) {
        Some(_) => i32_value(table, key),
        None => Ok(default),
    }
}

fn optional_number(table: &Table, key: &str) -> Result<Option<f64>, io::Error> {
    match table.get(key) {
        None => Ok(None),
        Some(Value::Float(value)) => Ok(Some(*value)),
        Some(Value::Integer(value)) => Ok(Some(*value as f64)),
        Some(_) => Err(invalid_data(format!("{key} must be numeric"))),
    }
}

fn float_value(table: &Table, key: &str) -> Result<f64, io::Error> {
    optional_number(table, key)?.ok_or_else(|| invalid_data(format!("{key} is required")))
}

fn float_value_default(table: &Table, key: &str, default: f64) -> Result<f64, io::Error> {
    Ok(optional_number(table, key)?.unwrap_or(default))
}

fn optional_seed(table: &Table) -> Option<u64> {
    table
        .get("seed")
        .and_then(Value::as_integer)
        .map(|seed| seed as u64)
}

fn set_common_kind(
    common: &mut Option<GeneticCodeKind>,
    current: GeneticCodeKind,
) -> Result<(), io::Error> {
    if common.is_some_and(|kind| kind != current) {
        return Err(invalid_data(
            "checkpoint automata use mixed genetic-code representations",
        ));
    }
    *common = Some(current);
    Ok(())
}

// ---------------------------------------------------------------------------
// TOML builder
// ---------------------------------------------------------------------------

fn build_toml(pop: &Population, maze: &Maze) -> Table {
    let cfg = &pop.config;

    // [meta]
    let mut meta = Table::new();
    meta.insert("class".into(), Value::String("Population".into()));
    meta.insert("schema_version".into(), Value::Integer(1));
    meta.insert("generation".into(), Value::Integer(pop.generation as i64));
    meta.insert("tick_count".into(), Value::Integer(pop.tick_count as i64));
    meta.insert(
        "automaton_class".into(),
        Value::String("MazeAutomaton".into()),
    );

    // [environment]
    let mut env = Table::new();
    env.insert("class".into(), Value::String("Maze".into()));
    env.insert("name".into(), Value::String(maze.name.clone()));

    // [config]  (checkpoint config)
    let ckpt_base = "runs";
    let config_table = CheckpointConfig {
        enabled: false,
        generation_interval: 0,
    }
    .to_toml_table(ckpt_base);

    // [automaton_params]
    let mut automaton_params = Table::new();
    automaton_params.insert("state_bits".into(), Value::Integer(cfg.state_bits as i64));

    // [[fitness_history]]
    let fitness_history: Vec<Value> = pop
        .fitness_history
        .iter()
        .map(|s| {
            let mut t = Table::new();
            t.insert("generation".into(), Value::Integer(s.generation as i64));
            t.insert("min_fitness".into(), Value::Float(s.min_fitness));
            t.insert("max_fitness".into(), Value::Float(s.max_fitness));
            t.insert("mean_fitness".into(), Value::Float(s.mean_fitness));
            t.insert("duration_s".into(), Value::Float(s.duration_s));
            Value::Table(t)
        })
        .collect();

    // [[automata]]
    let automata: Vec<Value> = pop
        .automata
        .iter()
        .map(|a| {
            // genetic_code subtable
            let gc = &a.genetic_code;
            let mut gc_table = Table::new();
            gc_table.insert("type".into(), Value::String(gc.code_type().into()));
            gc_table.insert("schema_version".into(), Value::Integer(1));
            gc_table.insert("resp_bits".into(), Value::Integer(gc.resp_bits() as i64));
            if let Some(seed) = gc.code_seed() {
                gc_table.insert("seed".into(), Value::Integer(seed as i64));
            }
            if let Some(tsetlin) = gc.as_tsetlin() {
                gc_table.insert(
                    "num_clauses".into(),
                    Value::Integer(tsetlin.num_clauses() as i64),
                );
                gc_table.insert(
                    "input_bits".into(),
                    Value::Integer(tsetlin.input_bits() as i64),
                );
                gc_table.insert("threshold".into(), Value::Float(tsetlin.threshold()));
            }

            let mut at = Table::new();
            at.insert("name".into(), Value::String("Terry-2".into()));
            at.insert("fitness".into(), Value::Float(a.fitness));
            at.insert(
                "coords".into(),
                Value::Array(vec![
                    Value::Integer(a.x as i64),
                    Value::Integer(a.y as i64),
                    Value::Integer(a.orientation as i64),
                ]),
            );
            at.insert("last_action".into(), Value::Integer(a.last_action as i64));
            at.insert("env_bits".into(), Value::Integer(a.env_bits as i64));
            at.insert("state_bits".into(), Value::Integer(a.state_bits as i64));
            at.insert("resp_bits".into(), Value::Integer(a.resp_bits as i64));
            at.insert(
                "internal_state".into(),
                Value::Integer(a.internal_state as i64),
            );
            at.insert("energy".into(), Value::Integer(a.energy as i64));
            if let Some(fingerprint) = &a.fingerprint {
                at.insert(
                    "fingerprint_bits".into(),
                    Value::Integer(fingerprint.bits() as i64),
                );
                at.insert(
                    "fingerprint_value".into(),
                    Value::Integer(fingerprint.value() as i64),
                );
            }
            at.insert("genetic_code".into(), Value::Table(gc_table));
            Value::Table(at)
        })
        .collect();

    // Assemble root table.
    let mut root = Table::new();
    root.insert("meta".into(), Value::Table(meta));
    root.insert("environment".into(), Value::Table(env));
    root.insert("config".into(), Value::Table(config_table));
    if let Some(fingerprint) = &cfg.fingerprint {
        let mut fingerprint_config = Table::new();
        fingerprint_config.insert("bits".into(), Value::Integer(fingerprint.bits as i64));
        fingerprint_config.insert(
            "tournament_k".into(),
            Value::Integer(fingerprint.tournament_k as i64),
        );
        fingerprint_config.insert(
            "mutation_rate".into(),
            Value::Float(fingerprint.mutation_rate),
        );
        root.insert(
            "fingerprint_config".into(),
            Value::Table(fingerprint_config),
        );
    }
    root.insert("automaton_params".into(), Value::Table(automaton_params));
    root.insert("fitness_history".into(), Value::Array(fitness_history));
    root.insert("automata".into(), Value::Array(automata));
    root
}

fn build_wiki_toml(pop: &WikiPopulation, environment: &WikiEnvironment) -> Table {
    let cfg = &pop.config;
    let mut meta = Table::new();
    meta.insert("class".into(), Value::String("Population".into()));
    meta.insert("schema_version".into(), Value::Integer(1));
    meta.insert("generation".into(), Value::Integer(pop.generation as i64));
    meta.insert("tick_count".into(), Value::Integer(pop.tick_count as i64));
    meta.insert(
        "automaton_class".into(),
        Value::String("WikiAutomaton".into()),
    );

    let mut env = Table::new();
    env.insert("class".into(), Value::String("ByteEnv".into()));
    env.insert("name".into(), Value::String(environment.name.clone()));

    let config_table = CheckpointConfig {
        enabled: false,
        generation_interval: 0,
    }
    .to_toml_table("runs");

    let mut automaton_params = Table::new();
    automaton_params.insert("env_bits".into(), Value::Integer(16));
    automaton_params.insert("state_bits".into(), Value::Integer(cfg.state_bits as i64));
    automaton_params.insert("resp_bits".into(), Value::Integer(8));
    automaton_params.insert(
        "num_clauses".into(),
        Value::Integer(cfg.genetic_code.tsetlin_clauses as i64),
    );

    let fitness_history: Vec<Value> = pop
        .fitness_history
        .iter()
        .map(|stats| {
            let mut table = Table::new();
            table.insert("generation".into(), Value::Integer(stats.generation as i64));
            table.insert("min_fitness".into(), Value::Float(stats.min_fitness));
            table.insert("max_fitness".into(), Value::Float(stats.max_fitness));
            table.insert("mean_fitness".into(), Value::Float(stats.mean_fitness));
            table.insert("duration_s".into(), Value::Float(stats.duration_s));
            Value::Table(table)
        })
        .collect();

    let automata: Vec<Value> = pop
        .automata
        .iter()
        .map(|automaton| {
            let code = &automaton.genetic_code;
            let tsetlin = code
                .as_tsetlin()
                .expect("Wiki checkpoints require GeneticCodeTsetlin");
            let mut genetic_code = Table::new();
            genetic_code.insert("type".into(), Value::String(code.code_type().into()));
            genetic_code.insert("schema_version".into(), Value::Integer(1));
            genetic_code.insert("resp_bits".into(), Value::Integer(code.resp_bits() as i64));
            genetic_code.insert(
                "num_clauses".into(),
                Value::Integer(tsetlin.num_clauses() as i64),
            );
            genetic_code.insert(
                "input_bits".into(),
                Value::Integer(tsetlin.input_bits() as i64),
            );
            genetic_code.insert("threshold".into(), Value::Float(tsetlin.threshold()));
            if let Some(seed) = code.code_seed() {
                genetic_code.insert("seed".into(), Value::Integer(seed as i64));
            }

            let mut table = Table::new();
            table.insert("name".into(), Value::String("Terry-2".into()));
            table.insert("fitness".into(), Value::Float(automaton.fitness));
            table.insert(
                "coords".into(),
                Value::Array(vec![
                    Value::Integer(automaton.text_index as i64),
                    Value::Integer(automaton.byte_index as i64),
                ]),
            );
            table.insert(
                "last_action".into(),
                Value::Integer(automaton.last_action as i64),
            );
            table.insert("env_bits".into(), Value::Integer(16));
            table.insert(
                "state_bits".into(),
                Value::Integer(automaton.state_bits as i64),
            );
            table.insert("resp_bits".into(), Value::Integer(8));
            table.insert(
                "internal_state".into(),
                Value::Integer(automaton.internal_state as i64),
            );
            if let Some(fingerprint) = &automaton.fingerprint {
                table.insert(
                    "fingerprint_bits".into(),
                    Value::Integer(fingerprint.bits() as i64),
                );
                table.insert(
                    "fingerprint_value".into(),
                    Value::Integer(fingerprint.value() as i64),
                );
            }
            table.insert("genetic_code".into(), Value::Table(genetic_code));
            Value::Table(table)
        })
        .collect();

    let mut root = Table::new();
    root.insert("meta".into(), Value::Table(meta));
    root.insert("environment".into(), Value::Table(env));
    root.insert("config".into(), Value::Table(config_table));
    if let Some(fingerprint) = &cfg.fingerprint {
        let mut fingerprint_config = Table::new();
        fingerprint_config.insert("bits".into(), Value::Integer(fingerprint.bits as i64));
        fingerprint_config.insert(
            "tournament_k".into(),
            Value::Integer(fingerprint.tournament_k as i64),
        );
        fingerprint_config.insert(
            "mutation_rate".into(),
            Value::Float(fingerprint.mutation_rate),
        );
        root.insert(
            "fingerprint_config".into(),
            Value::Table(fingerprint_config),
        );
    }
    root.insert("automaton_params".into(), Value::Table(automaton_params));
    root.insert("fitness_history".into(), Value::Array(fitness_history));
    root.insert("automata".into(), Value::Array(automata));
    root
}

// ---------------------------------------------------------------------------
// NPZ builder
// ---------------------------------------------------------------------------

fn build_npz(pop: &Population) -> io::Result<Vec<u8>> {
    use zip::write::{FileOptions, ZipWriter};
    use zip::CompressionMethod;

    let buf = Vec::new();
    let cursor = std::io::Cursor::new(buf);
    let mut zip = ZipWriter::new(cursor);

    let options: FileOptions<()> =
        FileOptions::default().compression_method(CompressionMethod::Deflated);

    // Per-automaton arrays.
    for (i, a) in pop.automata.iter().enumerate() {
        match &a.genetic_code {
            GeneticCode::Dict(_) => {
                let entries = a.genetic_code.entries();
                let keys: Vec<i64> = entries.iter().map(|(key, _)| *key as i64).collect();
                let values: Vec<i64> = entries.iter().map(|(_, value)| *value as i64).collect();
                zip.start_file(format!("automaton_{i}_keys.npy"), options)?;
                write_npy_i64(&mut zip, &keys)?;
                zip.start_file(format!("automaton_{i}_values.npy"), options)?;
                write_npy_i64(&mut zip, &values)?;
            }
            GeneticCode::List(_) => {
                let values: Vec<i64> = a
                    .genetic_code
                    .entries()
                    .into_iter()
                    .map(|(_, value)| value as i64)
                    .collect();
                zip.start_file(format!("automaton_{i}_values.npy"), options)?;
                write_npy_i64(&mut zip, &values)?;
            }
            GeneticCode::Tsetlin(code) => {
                zip.start_file(format!("automaton_{i}_w_pos.npy"), options)?;
                write_npy_u64_2d(
                    &mut zip,
                    code.positive_masks(),
                    code.output_bits() as usize,
                    code.num_clauses(),
                )?;
                zip.start_file(format!("automaton_{i}_w_neg.npy"), options)?;
                write_npy_u64_2d(
                    &mut zip,
                    code.negative_masks(),
                    code.output_bits() as usize,
                    code.num_clauses(),
                )?;
            }
        }

        // automaton_{i}_energy_grid
        zip.start_file(format!("automaton_{i}_energy_grid.npy"), options)?;
        write_npy_u8(&mut zip, &a.energy_grid)?;
    }

    // fitness_history_fitnesses  (shape = [generations, pop_size])
    if !pop.fitness_history.is_empty() {
        let n_gens = pop.fitness_history.len();
        let pop_size = pop.fitness_history[0].fitnesses.len();
        let flat: Vec<f64> = pop
            .fitness_history
            .iter()
            .flat_map(|s| s.fitnesses.iter().cloned())
            .collect();
        zip.start_file("fitness_history_fitnesses.npy", options)?;
        write_npy_f64_2d(&mut zip, &flat, n_gens, pop_size)?;
    }

    let cursor = zip.finish()?;
    Ok(cursor.into_inner())
}

fn build_wiki_npz(pop: &WikiPopulation) -> io::Result<Vec<u8>> {
    use zip::write::{FileOptions, ZipWriter};
    use zip::CompressionMethod;

    let cursor = std::io::Cursor::new(Vec::new());
    let mut zip = ZipWriter::new(cursor);
    let options: FileOptions<()> =
        FileOptions::default().compression_method(CompressionMethod::Deflated);

    for (index, automaton) in pop.automata.iter().enumerate() {
        let code = automaton
            .genetic_code
            .as_tsetlin()
            .expect("Wiki checkpoints require GeneticCodeTsetlin");
        zip.start_file(format!("automaton_{index}_w_pos.npy"), options)?;
        write_npy_u64_2d(
            &mut zip,
            code.positive_masks(),
            code.output_bits() as usize,
            code.num_clauses(),
        )?;
        zip.start_file(format!("automaton_{index}_w_neg.npy"), options)?;
        write_npy_u64_2d(
            &mut zip,
            code.negative_masks(),
            code.output_bits() as usize,
            code.num_clauses(),
        )?;
    }

    if !pop.fitness_history.is_empty() {
        let rows = pop.fitness_history.len();
        let columns = pop.fitness_history[0].fitnesses.len();
        let values: Vec<f64> = pop
            .fitness_history
            .iter()
            .flat_map(|stats| stats.fitnesses.iter().copied())
            .collect();
        zip.start_file("fitness_history_fitnesses.npy", options)?;
        write_npy_f64_2d(&mut zip, &values, rows, columns)?;
    }

    Ok(zip.finish()?.into_inner())
}

// ---------------------------------------------------------------------------
// NPY helpers
// ---------------------------------------------------------------------------
//
// NPY v1.0 format:
//   magic   6 bytes  \x93NUMPY
//   version 2 bytes  \x01\x00
//   hdrlen  2 bytes  uint16 little-endian
//   header  hdrlen bytes  ASCII dict + padding spaces + \n
//   data    raw little-endian bytes
//
// Total preamble (10 + hdrlen) must be a multiple of 64.

fn write_npy_header(w: &mut impl Write, descr: &str, shape_str: &str) -> io::Result<()> {
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}");
    // preamble = 10 bytes (magic + version + hdrlen field)
    let preamble = 10usize;
    let dict_bytes = dict.len();
    // We need preamble + hdrlen to be a multiple of 64, and hdrlen = dict_bytes + padding + 1 (\n)
    let min_hdrlen = dict_bytes + 1; // at least dict + newline
    let hdrlen = (preamble + min_hdrlen).div_ceil(64) * 64 - preamble;
    let padding = hdrlen - dict_bytes - 1;

    w.write_all(b"\x93NUMPY")?;
    w.write_all(&[1u8, 0u8])?;
    w.write_all(&(hdrlen as u16).to_le_bytes())?;
    w.write_all(dict.as_bytes())?;
    for _ in 0..padding {
        w.write_all(b" ")?;
    }
    w.write_all(b"\n")?;
    Ok(())
}

fn write_npy_i64(w: &mut impl Write, data: &[i64]) -> io::Result<()> {
    let shape_str = format!("({},)", data.len());
    write_npy_header(w, "<i8", &shape_str)?;
    for &v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn write_npy_u8(w: &mut impl Write, data: &[u8]) -> io::Result<()> {
    let shape_str = format!("({},)", data.len());
    write_npy_header(w, "|u1", &shape_str)?;
    w.write_all(data)?;
    Ok(())
}

fn write_npy_u64_2d(w: &mut impl Write, data: &[u64], rows: usize, cols: usize) -> io::Result<()> {
    let shape_str = format!("({rows}, {cols})");
    write_npy_header(w, "<u8", &shape_str)?;
    for &value in data {
        w.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn write_npy_f64_2d(w: &mut impl Write, data: &[f64], rows: usize, cols: usize) -> io::Result<()> {
    let shape_str = format!("({rows}, {cols})");
    write_npy_header(w, "<f8", &shape_str)?;
    for &v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fingerprint::FingerprintConfig;
    use crate::genetic_code::GeneticCodeConfig;
    use crate::population::PopConfig;
    use crate::wiki::{WikiEnvironment, WikiPopulation};
    use std::process::Command;

    #[test]
    fn python_loads_rust_tsetlin_population_checkpoint() {
        let maze = Maze::new("compat-maze", 4, 7);
        let config = PopConfig {
            size: 2,
            state_bits: 4,
            ticks_per_restart: 1,
            restarts_per_gen: 1,
            checkpoint_interval: 0,
            mutation_rate: 0.01,
            genetic_code: GeneticCodeConfig::default(),
            fingerprint: Some(FingerprintConfig {
                bits: 4,
                tournament_k: 2,
                mutation_rate: 0.01,
            }),
        };
        let population = Population::new(&maze, config, 123);
        let directory =
            std::env::temp_dir().join(format!("terry-rust-checkpoint-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let stem = directory.join("population");
        save_population(&population, &maze, &stem).unwrap();

        let python = if Path::new(".venv/bin/python").exists() {
            ".venv/bin/python"
        } else {
            "python3"
        };
        let script = format!(
            r#"
from pathlib import Path
from arc3_agi.genetic_code import GeneticCodeTsetlin
from arc3_agi.maze import Maze, MazeAutomaton
from arc3_agi.population import Population

population = Population.load(
    Path({stem:?}),
    environment=Maze("compat-maze", side_length_bits=4, seed=7),
    AutomatonClass=MazeAutomaton,
)
assert len(population.automata) == 2
assert population._fingerprint_config.bits == 4
for automaton in population.automata:
    assert isinstance(automaton.genetic_code, GeneticCodeTsetlin)
    assert automaton.genetic_code._w_pos.shape == (6, 4)
    assert automaton.genetic_code._w_neg.shape == (6, 4)
    assert automaton.genetic_code.threshold == 3.0
    assert automaton.fingerprint is not None
"#,
            stem = stem.display().to_string()
        );
        let output = Command::new(python)
            .arg("-c")
            .arg(script)
            .output()
            .expect("run Python checkpoint compatibility check");
        std::fs::remove_dir_all(&directory).ok();
        assert!(
            output.status.success(),
            "Python failed to load Rust checkpoint:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn python_loads_rust_dict_and_list_population_checkpoints() {
        let maze = Maze::new("compat-legacy", 4, 9);
        let directory = std::env::temp_dir().join(format!(
            "terry-rust-legacy-checkpoints-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&directory).unwrap();

        for (name, kind) in [
            ("dict", GeneticCodeKind::Dict),
            ("list", GeneticCodeKind::List),
        ] {
            let config = PopConfig {
                size: 2,
                state_bits: 4,
                ticks_per_restart: 1,
                restarts_per_gen: 1,
                checkpoint_interval: 0,
                mutation_rate: 0.01,
                genetic_code: GeneticCodeConfig {
                    kind,
                    ..GeneticCodeConfig::default()
                },
                fingerprint: None,
            };
            let mut population = Population::new(&maze, config, 42);
            population.run_generation(&maze);
            let stem = directory.join(name);
            save_population(&population, &maze, &stem).unwrap();
        }

        let python = if Path::new(".venv/bin/python").exists() {
            ".venv/bin/python"
        } else {
            "python3"
        };
        let script = format!(
            r#"
from pathlib import Path
from arc3_agi.genetic_code import GeneticCodeDict, GeneticCodeList
from arc3_agi.maze import Maze, MazeAutomaton
from arc3_agi.population import Population

maze = Maze("compat-legacy", side_length_bits=4, seed=9)
for name, expected in (("dict", GeneticCodeDict), ("list", GeneticCodeList)):
    population = Population.load(
        Path({directory:?}) / name,
        environment=maze,
        AutomatonClass=MazeAutomaton,
    )
    assert all(isinstance(a.genetic_code, expected) for a in population.automata)
"#,
            directory = directory.display().to_string()
        );
        let output = Command::new(python)
            .arg("-c")
            .arg(script)
            .output()
            .expect("run Python legacy checkpoint compatibility check");
        std::fs::remove_dir_all(&directory).ok();
        assert!(
            output.status.success(),
            "Python failed to load Rust Dict/List checkpoints:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn rust_round_trips_tsetlin_population_checkpoint() {
        let maze = Maze::new("round-trip-maze", 4, 11);
        let config = PopConfig {
            size: 3,
            state_bits: 4,
            ticks_per_restart: 2,
            restarts_per_gen: 1,
            checkpoint_interval: 0,
            mutation_rate: 0.02,
            genetic_code: GeneticCodeConfig::default(),
            fingerprint: Some(FingerprintConfig {
                bits: 4,
                tournament_k: 2,
                mutation_rate: 0.03,
            }),
        };
        let population = Population::new(&maze, config, 77);
        let directory =
            std::env::temp_dir().join(format!("terry-rust-round-trip-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let stem = directory.join("population");
        save_population(&population, &maze, &stem).unwrap();

        let restored = load_population(
            &stem,
            &maze,
            &ResumeConfig {
                ticks_per_restart: 2,
                restarts_per_gen: 1,
                checkpoint_interval: 0,
                mutation_rate: 0.02,
                seed: 77,
            },
        )
        .unwrap();
        std::fs::remove_dir_all(&directory).ok();

        assert_eq!(restored.generation, 0);
        assert_eq!(restored.automata.len(), 3);
        assert!(restored.config.fingerprint.is_some());
        for (original, loaded) in population.automata.iter().zip(&restored.automata) {
            let original = original.genetic_code.as_tsetlin().unwrap();
            let loaded = loaded.genetic_code.as_tsetlin().unwrap();
            assert_eq!(original.positive_masks(), loaded.positive_masks());
            assert_eq!(original.negative_masks(), loaded.negative_masks());
            assert_eq!(original.threshold(), loaded.threshold());
            assert_eq!(original.num_clauses(), loaded.num_clauses());
        }
    }

    #[test]
    fn python_loads_rust_wiki_population_checkpoint() {
        let environment = WikiEnvironment::new("WikiEnv", vec![b"abc".to_vec()]).unwrap();
        let config = PopConfig {
            size: 2,
            state_bits: 8,
            ticks_per_restart: 3,
            restarts_per_gen: 1,
            checkpoint_interval: 0,
            mutation_rate: 0.01,
            genetic_code: GeneticCodeConfig {
                kind: GeneticCodeKind::Tsetlin,
                tsetlin_clauses: 2,
            },
            fingerprint: Some(FingerprintConfig {
                bits: 4,
                tournament_k: 2,
                mutation_rate: 0.01,
            }),
        };
        let mut population = WikiPopulation::new(&environment, config, 123);
        population.run_generation(&environment);
        population.evolve(&environment);
        let directory =
            std::env::temp_dir().join(format!("terry-rust-wiki-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let stem = directory.join("population");
        save_wiki_population(&population, &environment, &stem).unwrap();

        let python = if Path::new(".venv/bin/python").exists() {
            ".venv/bin/python"
        } else {
            "python3"
        };
        let script = format!(
            r#"
from pathlib import Path
from arc3_agi.environment import ByteEnv
from arc3_agi.genetic_code import GeneticCodeTsetlin
from arc3_agi.population import Population
from arc3_agi.wiki_text_2 import WikiAutomaton

environment = ByteEnv(name="WikiEnv", array=["abc"])
population = Population.load(
    Path({stem:?}),
    environment=environment,
    AutomatonClass=WikiAutomaton,
)
assert population.generation == 1
assert len(population.automata) == 2
assert population._fingerprint_config.bits == 4
for automaton in population.automata:
    assert isinstance(automaton.genetic_code, GeneticCodeTsetlin)
    assert automaton.genetic_code._w_pos.shape == (16, 2)
    assert automaton.genetic_code._w_neg.shape == (16, 2)
    assert automaton.fingerprint is not None
"#,
            stem = stem.display().to_string()
        );
        let output = Command::new(python)
            .arg("-c")
            .arg(script)
            .output()
            .expect("run Python Wiki checkpoint compatibility check");
        std::fs::remove_dir_all(&directory).ok();
        assert!(
            output.status.success(),
            "Python failed to load Rust Wiki checkpoint:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn rust_loads_python_wiki_population_checkpoint() {
        let directory =
            std::env::temp_dir().join(format!("terry-python-wiki-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let stem = directory.join("population");
        let python = if Path::new(".venv/bin/python").exists() {
            ".venv/bin/python"
        } else {
            "python3"
        };
        let script = format!(
            r#"
from pathlib import Path
from arc3_agi.checkpoint import CheckpointConfig
from arc3_agi.environment import ByteEnv
from arc3_agi.fingerprint import FingerprintConfig
from arc3_agi.population import Population
from arc3_agi.wiki_text_2 import WikiAutomaton

environment = ByteEnv(name="WikiEnv", array=["abc"])
population = Population(
    size=2,
    AutomatonClass=WikiAutomaton,
    environment=environment,
    seed=123,
    automaton_params={{"state_bits": 8, "num_clauses": 2}},
    checkpoint_config=CheckpointConfig(enabled=False),
    fingerprint_config=FingerprintConfig(bits=4, tournament_k=2),
)
population.run_generation(3)
population.evolve()
population.save(Path({stem:?}))
"#,
            stem = stem.display().to_string()
        );
        let output = Command::new(python)
            .arg("-c")
            .arg(script)
            .output()
            .expect("generate Python Wiki checkpoint");
        assert!(
            output.status.success(),
            "Python failed to save Wiki checkpoint:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );

        let environment = WikiEnvironment::new("WikiEnv", vec![b"abc".to_vec()]).unwrap();
        let population = load_wiki_population(
            &stem,
            &environment,
            &ResumeConfig {
                ticks_per_restart: 3,
                restarts_per_gen: 1,
                checkpoint_interval: 0,
                mutation_rate: 0.01,
                seed: 123,
            },
        )
        .unwrap();
        std::fs::remove_dir_all(&directory).ok();

        assert_eq!(population.generation, 1);
        assert_eq!(population.automata.len(), 2);
        assert_eq!(population.config.state_bits, 8);
        assert_eq!(population.config.genetic_code.tsetlin_clauses, 2);
        assert!(population
            .automata
            .iter()
            .all(|automaton| automaton.fingerprint.is_some()));
    }

    #[test]
    fn rust_infers_legacy_tsetlin_dimensions_from_arrays() {
        let maze = Maze::new("legacy-tsetlin", 4, 13);
        let config = PopConfig {
            size: 2,
            state_bits: 4,
            ticks_per_restart: 1,
            restarts_per_gen: 1,
            checkpoint_interval: 0,
            mutation_rate: 0.01,
            genetic_code: GeneticCodeConfig {
                tsetlin_clauses: 5,
                ..GeneticCodeConfig::default()
            },
            fingerprint: None,
        };
        let population = Population::new(&maze, config, 88);
        let directory =
            std::env::temp_dir().join(format!("terry-rust-legacy-tsetlin-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let stem = directory.join("population");
        save_population(&population, &maze, &stem).unwrap();

        let toml_path = stem.with_extension("toml");
        let mut document: Value = toml::from_str(&std::fs::read_to_string(&toml_path).unwrap())
            .expect("parse generated checkpoint");
        for automaton in document
            .as_table_mut()
            .unwrap()
            .get_mut("automata")
            .unwrap()
            .as_array_mut()
            .unwrap()
        {
            let genetic_code = automaton
                .as_table_mut()
                .unwrap()
                .get_mut("genetic_code")
                .unwrap()
                .as_table_mut()
                .unwrap();
            genetic_code.remove("resp_bits");
            genetic_code.remove("num_clauses");
            genetic_code.remove("input_bits");
            genetic_code.remove("threshold");
        }
        std::fs::write(&toml_path, toml::to_string(&document).unwrap()).unwrap();

        let restored = load_population(
            &stem,
            &maze,
            &ResumeConfig {
                ticks_per_restart: 1,
                restarts_per_gen: 1,
                checkpoint_interval: 0,
                mutation_rate: 0.01,
                seed: 88,
            },
        )
        .unwrap();
        std::fs::remove_dir_all(&directory).ok();

        let code = restored.automata[0].genetic_code.as_tsetlin().unwrap();
        assert_eq!(code.output_bits(), 6);
        assert_eq!(code.num_clauses(), 5);
        assert_eq!(code.input_bits(), 64);
        assert_eq!(code.threshold(), 3.0);
    }

    #[test]
    fn rust_loads_existing_python_tsetlin_checkpoint() {
        let stem = Path::new("runs/2026-07-25T20-01-25-932326/gen_000800");
        if !stem.with_extension("toml").exists() {
            return;
        }
        let maze = Maze::new("ExampleMaze", 6, 42);
        let population = load_population(
            stem,
            &maze,
            &ResumeConfig {
                ticks_per_restart: 1,
                restarts_per_gen: 1,
                checkpoint_interval: 0,
                mutation_rate: 0.01,
                seed: 1234,
            },
        )
        .unwrap();

        assert_eq!(population.generation, 800);
        assert_eq!(population.automata.len(), 100);
        assert_eq!(population.fitness_history.len(), 800);
        assert_eq!(population.config.fingerprint.as_ref().unwrap().bits, 4);
        assert!(population
            .automata
            .iter()
            .all(|automaton| automaton.genetic_code.as_tsetlin().is_some()));
        assert!(population
            .automata
            .iter()
            .all(|automaton| automaton.fingerprint.is_some()));
    }
}
