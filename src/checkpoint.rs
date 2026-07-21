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
use std::io::{self, Write};
use std::path::Path;

use toml::{Table, Value};

use crate::maze::Maze;
use crate::population::Population;

/// Checkpoint configuration.
#[derive(Clone, Debug)]
pub struct CheckpointConfig {
    pub enabled: bool,
    pub generation_interval: usize,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        CheckpointConfig { enabled: false, generation_interval: 0 }
    }
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
    let toml_str = toml::to_string(&toml_doc)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
    std::fs::write(&toml_path, toml_str.as_bytes())?;

    let npz_bytes = build_npz(population)?;
    std::fs::write(&npz_path, &npz_bytes)?;

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
    meta.insert("automaton_class".into(), Value::String("MazeAutomaton".into()));

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
            let gc = a.genetic_code.as_ref();
            let mut gc_table = Table::new();
            gc_table.insert("type".into(), Value::String(gc.code_type().into()));
            gc_table.insert("schema_version".into(), Value::Integer(1));
            gc_table.insert("resp_bits".into(), Value::Integer(gc.resp_bits() as i64));
            if let Some(seed) = gc.code_seed() {
                gc_table.insert("seed".into(), Value::Integer(seed as i64));
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
            at.insert("genetic_code".into(), Value::Table(gc_table));
            Value::Table(at)
        })
        .collect();

    // Assemble root table.
    let mut root = Table::new();
    root.insert("meta".into(), Value::Table(meta));
    root.insert("environment".into(), Value::Table(env));
    root.insert("config".into(), Value::Table(config_table));
    root.insert("automaton_params".into(), Value::Table(automaton_params));
    root.insert(
        "fitness_history".into(),
        Value::Array(fitness_history),
    );
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

    let options: FileOptions<()> = FileOptions::default()
        .compression_method(CompressionMethod::Deflated);

    // Per-automaton arrays.
    for (i, a) in pop.automata.iter().enumerate() {
        let entries = a.genetic_code.entries();
        let mut keys_i64: Vec<i64> = Vec::with_capacity(entries.len());
        let mut values_i64: Vec<i64> = Vec::with_capacity(entries.len());
        for (k, v) in &entries {
            keys_i64.push(*k as i64);
            values_i64.push(*v as i64);
        }

        // automaton_{i}_keys
        zip.start_file(format!("automaton_{i}_keys.npy"), options)?;
        write_npy_i64(&mut zip, &keys_i64)?;

        // automaton_{i}_values
        zip.start_file(format!("automaton_{i}_values.npy"), options)?;
        write_npy_i64(&mut zip, &values_i64)?;

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
    let dict = format!(
        "{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}"
    );
    // preamble = 10 bytes (magic + version + hdrlen field)
    let preamble = 10usize;
    let dict_bytes = dict.len();
    // We need preamble + hdrlen to be a multiple of 64, and hdrlen = dict_bytes + padding + 1 (\n)
    let min_hdrlen = dict_bytes + 1; // at least dict + newline
    let hdrlen = ((preamble + min_hdrlen + 63) / 64) * 64 - preamble;
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

fn write_npy_f64_2d(
    w: &mut impl Write,
    data: &[f64],
    rows: usize,
    cols: usize,
) -> io::Result<()> {
    let shape_str = format!("({rows}, {cols})");
    write_npy_header(w, "<f8", &shape_str)?;
    for &v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}
