use std::error::Error;
use std::fs::File;
use std::io::copy;
use std::path::Path;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow_array::{Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rand::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::fingerprint::SelectionFingerprint;
use crate::genetic_code::{GeneticCode, GeneticCodeConfig, GeneticCodeKind};
use crate::population::{PopulationAutomaton, PopulationCore};

pub const WIKITEXT_DATASET_NAME: &str = "Salesforce/wikitext";
pub const WIKITEXT_DATASET_CONFIG: &str = "wikitext-2-raw-v1";
pub const WIKITEXT_SPLIT: &str = "train";
const HUGGING_FACE_DATASETS_URL: &str = "https://huggingface.co/datasets";
const WIKI_RESET_SEQUENCE_SEED: u64 = 0x5EED_5EED_5EED_5EED;

pub type WikiResult<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Debug)]
pub struct WikiEnvironment {
    pub name: String,
    texts: Vec<Vec<u8>>,
}

#[derive(Clone, Debug)]
pub struct LoadedWikiEnvironment {
    pub environment: WikiEnvironment,
    pub source_path: PathBuf,
}

pub struct WikiAutomaton {
    pub id: u64,
    pub text_index: usize,
    pub byte_index: usize,
    pub remaining_bytes: usize,
    pub internal_state: u16,
    pub state_bits: u8,
    state_mask: u16,
    pub right: u64,
    pub total: u64,
    pub fitness: f64,
    pub last_action: i32,
    pub genetic_code: GeneticCode,
    pub fingerprint: Option<SelectionFingerprint>,
    rng: Xoshiro256PlusPlus,
}

pub type WikiPopulation = PopulationCore<WikiAutomaton>;

impl WikiAutomaton {
    pub const ENV_BITS: u8 = 16;
    pub const RESPONSE_BITS: u8 = 8;

    pub fn new(
        environment: &WikiEnvironment,
        state_bits: u8,
        code_config: &GeneticCodeConfig,
        seed: u64,
    ) -> Result<Self, String> {
        if code_config.kind != GeneticCodeKind::Tsetlin {
            return Err("WikiAutomaton requires GeneticCodeTsetlin".into());
        }
        validate_state_bits(state_bits)?;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let genetic_code = GeneticCode::new(
            code_config,
            state_bits + Self::RESPONSE_BITS,
            state_bits + Self::ENV_BITS,
            rng.next_u32() as u64,
        )?;
        Ok(Self::from_parts(
            genetic_code,
            environment,
            state_bits,
            &mut rng,
        ))
    }

    pub fn with_code(
        genetic_code: GeneticCode,
        environment: &WikiEnvironment,
        state_bits: u8,
        seed: u64,
    ) -> Result<Self, String> {
        validate_state_bits(state_bits)?;
        if genetic_code.resp_bits() != state_bits + Self::RESPONSE_BITS {
            return Err("genetic-code output width does not match WikiAutomaton".into());
        }
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        Ok(Self::from_parts(
            genetic_code,
            environment,
            state_bits,
            &mut rng,
        ))
    }

    fn from_parts(
        genetic_code: GeneticCode,
        environment: &WikiEnvironment,
        state_bits: u8,
        _rng: &mut Xoshiro256PlusPlus,
    ) -> Self {
        debug_assert!(!environment.texts().is_empty());
        Self {
            id: 0,
            text_index: 0,
            byte_index: 0,
            remaining_bytes: 0,
            internal_state: 0,
            state_bits,
            state_mask: (1u16 << state_bits) - 1,
            right: 0,
            total: 0,
            fitness: 0.0,
            last_action: -1,
            genetic_code,
            fingerprint: None,
            // Keep reset targets random over time, but identical at each reset
            // step for all automata so fitness is comparable.
            rng: Xoshiro256PlusPlus::seed_from_u64(WIKI_RESET_SEQUENCE_SEED),
        }
    }

    pub fn tick(&mut self, environment: &WikiEnvironment) -> u8 {
        if self.remaining_bytes == 0 {
            self.text_index = (self.text_index + 1) % environment.texts().len();
            self.byte_index = 0;
            self.remaining_bytes = environment.texts()[self.text_index].len();
        }

        let observation = environment.observation(self.text_index, self.byte_index);
        let input_code = ((self.internal_state as u32) << Self::ENV_BITS) | observation as u32;
        let output_code = self.genetic_code.get(input_code);
        self.internal_state = output_code & self.state_mask;
        let prediction = (output_code >> self.state_bits) as u8;

        self.total += 1;
        self.remaining_bytes -= 1;
        self.byte_index += 1;
        let actual = if self.remaining_bytes == 0 {
            0
        } else {
            environment.texts()[self.text_index][self.byte_index]
        };
        if prediction == actual {
            self.right += 1;
        }
        self.fitness = self.right as f64 / self.total as f64;
        prediction
    }

    pub fn reset(&mut self, environment: &WikiEnvironment) {
        self.text_index = (self.rng.next_u64() as usize) % environment.texts().len();
        self.byte_index = 0;
        self.remaining_bytes = 0;
        self.internal_state = 0;
        self.right = 0;
        self.total = 0;
        self.fitness = 0.0;
        self.last_action = -1;
    }

    pub fn is_active(&self) -> bool {
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub fn restore(
        genetic_code: GeneticCode,
        environment: &WikiEnvironment,
        state_bits: u8,
        seed: u64,
        text_index: usize,
        byte_index: usize,
        internal_state: u16,
        fitness: f64,
        last_action: i32,
        fingerprint: Option<SelectionFingerprint>,
    ) -> Result<Self, String> {
        if text_index >= environment.texts().len()
            || byte_index > environment.texts()[text_index].len()
        {
            return Err("checkpoint automaton coordinates are outside WikiEnvironment".into());
        }
        let mut automaton = Self::with_code(genetic_code, environment, state_bits, seed)?;
        if internal_state > automaton.state_mask {
            return Err("checkpoint internal state is outside state_bits".into());
        }
        automaton.text_index = text_index;
        automaton.byte_index = byte_index;
        automaton.internal_state = internal_state;
        automaton.fitness = fitness;
        automaton.last_action = last_action;
        automaton.fingerprint = fingerprint;
        Ok(automaton)
    }
}

fn validate_state_bits(state_bits: u8) -> Result<(), String> {
    if !(1..=8).contains(&state_bits) {
        return Err("WikiAutomaton state_bits must be between 1 and 8".into());
    }
    Ok(())
}

impl PopulationAutomaton for WikiAutomaton {
    type Environment = WikiEnvironment;

    fn new(
        environment: &WikiEnvironment,
        state_bits: u8,
        code_config: &GeneticCodeConfig,
        seed: u64,
    ) -> Result<Self, String> {
        WikiAutomaton::new(environment, state_bits, code_config, seed)
    }

    fn with_code(
        genetic_code: GeneticCode,
        environment: &WikiEnvironment,
        state_bits: u8,
        seed: u64,
    ) -> Self {
        WikiAutomaton::with_code(genetic_code, environment, state_bits, seed)
            .expect("invalid inherited Wiki genetic code")
    }

    fn tick(&mut self, environment: &WikiEnvironment) {
        WikiAutomaton::tick(self, environment);
    }

    fn reset(&mut self, environment: &WikiEnvironment) {
        WikiAutomaton::reset(self, environment);
    }

    fn is_active(&self) -> bool {
        WikiAutomaton::is_active(self)
    }

    fn id(&self) -> u64 {
        self.id
    }

    fn set_id(&mut self, id: u64) {
        self.id = id;
    }

    fn fitness(&self) -> f64 {
        self.fitness
    }

    fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness;
    }

    fn genetic_code(&self) -> &GeneticCode {
        &self.genetic_code
    }

    fn fingerprint(&self) -> Option<&SelectionFingerprint> {
        self.fingerprint.as_ref()
    }

    fn fingerprint_mut(&mut self) -> &mut Option<SelectionFingerprint> {
        &mut self.fingerprint
    }
}

impl WikiEnvironment {
    pub fn new(name: impl Into<String>, texts: Vec<Vec<u8>>) -> WikiResult<Self> {
        let texts: Vec<Vec<u8>> = texts.into_iter().filter(|text| !text.is_empty()).collect();
        if texts.is_empty() {
            return Err("Wiki environment must contain at least one non-empty text".into());
        }
        Ok(Self {
            name: name.into(),
            texts,
        })
    }

    pub fn from_parquet(path: &Path) -> WikiResult<Self> {
        let file = File::open(path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let text_index = builder
            .schema()
            .fields()
            .iter()
            .position(|field| field.name() == "text")
            .ok_or("WikiText Parquet file is missing the text column")?;
        let reader = builder.build()?;
        let mut texts = Vec::new();

        for batch in reader {
            let batch = batch?;
            let column = batch
                .column(text_index)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or("WikiText Parquet text column must contain UTF-8 strings")?;
            for row in 0..column.len() {
                if column.is_null(row) {
                    return Err("WikiText Parquet text column contains a null value".into());
                }
                let text = column.value(row);
                if !text.is_empty() {
                    texts.push(text.as_bytes().to_vec());
                }
            }
        }

        Self::new("WikiEnv", texts)
    }

    pub fn texts(&self) -> &[Vec<u8>] {
        &self.texts
    }

    pub fn observation(&self, text_index: usize, byte_index: usize) -> u16 {
        let text = &self.texts[text_index];
        let current = text[byte_index] as u16;
        if byte_index == 0 {
            current
        } else {
            ((text[byte_index - 1] as u16) << 8) | current
        }
    }
}

pub fn load_wikitext_environment(
    dataset_name: &str,
    dataset_config: &str,
    split: &str,
    dataset_path: Option<&Path>,
    cache_root: Option<&Path>,
) -> WikiResult<LoadedWikiEnvironment> {
    load_wikitext_environment_from_base(
        dataset_name,
        dataset_config,
        split,
        dataset_path,
        cache_root,
        HUGGING_FACE_DATASETS_URL,
    )
}

fn load_wikitext_environment_from_base(
    dataset_name: &str,
    dataset_config: &str,
    split: &str,
    dataset_path: Option<&Path>,
    cache_root: Option<&Path>,
    base_url: &str,
) -> WikiResult<LoadedWikiEnvironment> {
    if let Some(path) = dataset_path {
        return Ok(LoadedWikiEnvironment {
            environment: WikiEnvironment::from_parquet(path)?,
            source_path: path.to_path_buf(),
        });
    }

    validate_dataset_name(dataset_name)?;
    validate_path_component(dataset_config, "dataset config")?;
    validate_path_component(split, "dataset split")?;
    let cache_root = match cache_root {
        Some(path) => path.to_path_buf(),
        None => default_cache_root()?,
    };
    let path = cache_root
        .join("terry-2")
        .join("wikitext")
        .join(dataset_config)
        .join(format!("{split}.parquet"));

    if path.exists() {
        match WikiEnvironment::from_parquet(&path) {
            Ok(environment) => {
                return Ok(LoadedWikiEnvironment {
                    environment,
                    source_path: path,
                });
            }
            Err(_) => std::fs::remove_file(&path)?,
        }
    }

    let url = dataset_url(base_url, dataset_name, dataset_config, split)?;
    download_validated_parquet(&url, &path)?;
    Ok(LoadedWikiEnvironment {
        environment: WikiEnvironment::from_parquet(&path)?,
        source_path: path,
    })
}

fn default_cache_root() -> WikiResult<PathBuf> {
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        return Ok(PathBuf::from(path));
    }
    let home = std::env::var_os("HOME").ok_or("HOME is not set; pass an explicit cache path")?;
    Ok(PathBuf::from(home).join(".cache"))
}

fn validate_dataset_name(dataset_name: &str) -> WikiResult<()> {
    let segments: Vec<&str> = dataset_name.split('/').collect();
    if segments.len() != 2 {
        return Err("dataset name must have the form owner/name".into());
    }
    for segment in segments {
        validate_path_component(segment, "dataset name")?;
    }
    Ok(())
}

fn validate_path_component(value: &str, label: &str) -> WikiResult<()> {
    if value.is_empty()
        || value == "."
        || value == ".."
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    {
        return Err(format!("invalid {label}: {value:?}").into());
    }
    Ok(())
}

fn dataset_url(
    base_url: &str,
    dataset_name: &str,
    dataset_config: &str,
    split: &str,
) -> WikiResult<reqwest::Url> {
    let mut url = reqwest::Url::parse(base_url)?;
    let mut segments = url
        .path_segments_mut()
        .map_err(|_| "dataset base URL cannot be a base")?;
    segments.pop_if_empty();
    for segment in dataset_name.split('/') {
        segments.push(segment);
    }
    segments
        .push("resolve")
        .push("main")
        .push(dataset_config)
        .push(&format!("{split}-00000-of-00001.parquet"));
    drop(segments);
    Ok(url)
}

fn download_validated_parquet(url: &reqwest::Url, destination: &Path) -> WikiResult<()> {
    let parent = destination
        .parent()
        .ok_or("WikiText cache destination has no parent directory")?;
    std::fs::create_dir_all(parent)?;
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let temporary =
        destination.with_extension(format!("parquet.tmp-{}-{nonce}", std::process::id()));

    let result = (|| -> WikiResult<()> {
        let mut response = reqwest::blocking::Client::new()
            .get(url.clone())
            .send()?
            .error_for_status()?;
        let mut file = File::create(&temporary)?;
        copy(&mut response, &mut file)?;
        file.sync_all()?;
        drop(file);
        WikiEnvironment::from_parquet(&temporary)?;
        std::fs::rename(&temporary, destination)?;
        Ok(())
    })();

    if result.is_err() {
        std::fs::remove_file(&temporary).ok();
    }
    result
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    use arrow_array::RecordBatch;
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    use crate::fingerprint::FingerprintConfig;
    use crate::population::PopConfig;

    use super::*;

    fn test_directory(label: &str) -> std::path::PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("terry-{label}-{}-{nonce}", std::process::id()))
    }

    fn write_text_fixture(path: &Path, values: Vec<Option<&str>>) {
        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, true)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(StringArray::from(values))],
        )
        .unwrap();
        let file = File::create(path).unwrap();
        let mut writer = ArrowWriter::try_new(file, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
    }

    #[test]
    fn parquet_loader_preserves_non_empty_utf8_text_bytes() {
        let directory = test_directory("wiki-parquet");
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("train.parquet");
        write_text_fixture(&path, vec![Some(""), Some("a\n"), Some("é")]);

        let environment = WikiEnvironment::from_parquet(&path).unwrap();

        std::fs::remove_dir_all(directory).ok();
        assert_eq!(environment.name, "WikiEnv");
        assert_eq!(
            environment.texts(),
            &[b"a\n".to_vec(), "é".as_bytes().to_vec()]
        );
        assert_eq!(environment.observation(0, 0), b'a' as u16);
        assert_eq!(
            environment.observation(0, 1),
            ((b'a' as u16) << 8) | b'\n' as u16
        );
        assert_eq!(environment.observation(1, 1), 0xc3a9);
    }

    #[test]
    fn loader_downloads_once_then_reuses_validated_cache() {
        let directory = test_directory("wiki-cache");
        std::fs::create_dir_all(&directory).unwrap();
        let fixture = directory.join("fixture.parquet");
        write_text_fixture(&fixture, vec![Some("cached\n")]);
        let fixture_bytes = std::fs::read(&fixture).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 2048];
            let _ = stream.read(&mut request).unwrap();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                fixture_bytes.len()
            )
            .unwrap();
            stream.write_all(&fixture_bytes).unwrap();
        });
        let base_url = format!("http://{address}");

        let first = load_wikitext_environment_from_base(
            WIKITEXT_DATASET_NAME,
            WIKITEXT_DATASET_CONFIG,
            WIKITEXT_SPLIT,
            None,
            Some(&directory),
            &base_url,
        )
        .unwrap();
        server.join().unwrap();
        let second = load_wikitext_environment_from_base(
            WIKITEXT_DATASET_NAME,
            WIKITEXT_DATASET_CONFIG,
            WIKITEXT_SPLIT,
            None,
            Some(&directory),
            "http://127.0.0.1:1",
        )
        .unwrap();

        std::fs::remove_dir_all(directory).ok();
        assert_eq!(first.environment.texts(), &[b"cached\n".to_vec()]);
        assert_eq!(second.environment.texts(), first.environment.texts());
        assert_eq!(second.source_path, first.source_path);
    }

    #[test]
    fn explicit_dataset_path_bypasses_download_and_cache() {
        let directory = test_directory("wiki-local");
        std::fs::create_dir_all(&directory).unwrap();
        let fixture = directory.join("local.parquet");
        write_text_fixture(&fixture, vec![Some("local")]);

        let loaded = load_wikitext_environment_from_base(
            "not/a/validated/name",
            "ignored/config",
            "ignored/split",
            Some(&fixture),
            None,
            "not a URL",
        )
        .unwrap();

        std::fs::remove_dir_all(directory).ok();
        assert_eq!(loaded.source_path, fixture);
        assert_eq!(loaded.environment.texts(), &[b"local".to_vec()]);
    }

    #[test]
    fn parquet_loader_rejects_an_all_empty_dataset() {
        let directory = test_directory("wiki-empty");
        std::fs::create_dir_all(&directory).unwrap();
        let fixture = directory.join("empty.parquet");
        write_text_fixture(&fixture, vec![Some(""), Some("")]);

        let error = WikiEnvironment::from_parquet(&fixture).unwrap_err();

        std::fs::remove_dir_all(directory).ok();
        assert!(error.to_string().contains("at least one non-empty text"));
    }

    #[test]
    fn invalid_download_does_not_leave_cache_or_temporary_file() {
        let directory = test_directory("wiki-invalid-download");
        std::fs::create_dir_all(&directory).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0u8; 2048];
            let _ = stream.read(&mut request).unwrap();
            let body = b"not parquet";
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            stream.write_all(body).unwrap();
        });

        let result = load_wikitext_environment_from_base(
            WIKITEXT_DATASET_NAME,
            WIKITEXT_DATASET_CONFIG,
            WIKITEXT_SPLIT,
            None,
            Some(&directory),
            &format!("http://{address}"),
        );
        server.join().unwrap();
        let cache_directory = directory
            .join("terry-2")
            .join("wikitext")
            .join(WIKITEXT_DATASET_CONFIG);
        let remaining_files = std::fs::read_dir(&cache_directory)
            .unwrap()
            .filter_map(Result::ok)
            .count();

        std::fs::remove_dir_all(directory).ok();
        assert!(result.is_err());
        assert_eq!(remaining_files, 0);
    }

    #[test]
    fn automaton_scores_next_raw_byte_and_end_sentinel() {
        let environment = WikiEnvironment::new("test", vec![b"abc".to_vec()]).unwrap();
        let genetic_code = GeneticCode::from_dict_entries(
            vec![
                (b'a' as u32, (b'b' as u16) << 8),
                (((b'a' as u32) << 8) | b'b' as u32, (b'c' as u16) << 8),
                (((b'b' as u32) << 8) | b'c' as u32, 0),
            ],
            16,
            Some(0),
        );
        let mut automaton = WikiAutomaton::with_code(genetic_code, &environment, 8, 0).unwrap();

        let predictions: Vec<u8> = (0..3).map(|_| automaton.tick(&environment)).collect();

        assert_eq!(predictions, vec![b'b', b'c', 0]);
        assert_eq!(automaton.right, 3);
        assert_eq!(automaton.total, 3);
        assert_eq!(automaton.fitness, 1.0);
    }

    #[test]
    fn tiny_wiki_population_runs_and_evolves() {
        let environment =
            WikiEnvironment::new("test", vec![b"abc".to_vec(), b"def".to_vec()]).unwrap();
        let config = PopConfig {
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
            fingerprint: Some(FingerprintConfig {
                bits: 4,
                tournament_k: 2,
                mutation_rate: 0.01,
            }),
        };
        let mut population = WikiPopulation::new(&environment, config, 0);

        population.run_generation(&environment);
        let stats = population.evolve(&environment);

        assert_eq!(stats.generation, 1);
        assert_eq!(stats.fitnesses.len(), 4);
        assert!(stats.fitnesses.iter().all(|fitness| fitness.is_finite()));
        assert!(population
            .automata
            .iter()
            .all(|automaton| automaton.fingerprint.is_some()));
    }

    #[test]
    fn reset_uses_the_same_text_sequence_for_all_automata() {
        let environment = WikiEnvironment::new(
            "test",
            vec![b"aaa".to_vec(), b"bbb".to_vec(), b"ccc".to_vec()],
        )
        .unwrap();
        let mut first = WikiAutomaton::with_code(
            GeneticCode::new(&GeneticCodeConfig::default(), 16, 24, 7).unwrap(),
            &environment,
            8,
            1,
        )
        .unwrap();
        let mut second = WikiAutomaton::with_code(
            GeneticCode::new(&GeneticCodeConfig::default(), 16, 24, 8).unwrap(),
            &environment,
            8,
            2,
        )
        .unwrap();

        for _ in 0..8 {
            first.reset(&environment);
            second.reset(&environment);
            assert_eq!(first.text_index, second.text_index);
        }
    }
}
