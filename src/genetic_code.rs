use rand::rngs::StdRng;
/// Genetic code representations.
///
/// Provides a `GeneticCode` trait with two implementations:
/// - `GeneticCodeDict`: sparse `HashMap`-backed with lazy fill (primary).
/// - `GeneticCodeList`: dense `Vec`-backed for contiguous key spaces.
///
/// Mirrors the Python `GeneticCode` / `GeneticCodeDict` / `GeneticCodeList`
/// hierarchy, keeping the same crossover algorithm and checkpoint semantics.
use rand::{RngCore, SeedableRng};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Core interface for a genetic code (input_code → output_code mapping).
///
/// Object-safe so it can be held as `Box<dyn GeneticCode>`.
pub trait GeneticCode: Send {
    /// Look up `key`; lazy-create a random value on miss (for Dict variant).
    fn get(&mut self, key: u32) -> u8;

    fn set(&mut self, key: u32, value: u8);

    /// Return all populated (key, value) pairs.  Used for crossover and NPZ
    /// serialisation.  For `GeneticCodeList` this is all indices.
    fn entries(&self) -> Vec<(u32, u8)>;

    /// Produce a child code by combining `self` (parent 1) with `other`
    /// (parent 2) and applying geometric-gap bit-flip mutation.
    fn crossover(
        &self,
        other: &dyn GeneticCode,
        mutation_rate: f64,
        rng: &mut dyn RngCore,
    ) -> Box<dyn GeneticCode>;

    /// Type tag written to TOML checkpoints (`"GeneticCodeDict"` or
    /// `"GeneticCodeList"`).
    fn code_type(&self) -> &'static str;

    /// Number of bits in the output code (= `state_bits + resp_bits`).
    fn resp_bits(&self) -> u8;

    /// Optional seed stored in the TOML checkpoint for reproducibility.
    fn code_seed(&self) -> Option<u64>;

    /// Deep-clone into a new `Box<dyn GeneticCode>`.
    fn clone_box(&self) -> Box<dyn GeneticCode>;
}

// ---------------------------------------------------------------------------
// GeneticCodeDict  (primary — sparse HashMap with lazy fill)
// ---------------------------------------------------------------------------

/// Sparse genetic code backed by a `HashMap<u32, u8>`.
///
/// Keys are generated on first access (lazy fill) using a seeded RNG so that
/// unseen state/environment combinations produce a random (but deterministic)
/// output rather than panicking.  This mirrors Python's `GeneticCodeDict`
/// exactly, including the on-miss insertion semantics.
pub struct GeneticCodeDict {
    map: HashMap<u32, u8>,
    /// Number of bits in the output value (`state_bits + resp_bits`).
    output_bits: u8,
    /// Mask derived from `output_bits`.
    output_mask: u8,
    /// Optional seed stored for checkpoint round-trips.
    seed: Option<u64>,
    /// RNG used for lazy-fill value generation.
    cold_rng: StdRng,
}

impl GeneticCodeDict {
    /// Create an empty dict pre-seeded with `seed`.
    pub fn new(output_bits: u8, seed: u64) -> Self {
        GeneticCodeDict {
            map: HashMap::new(),
            output_bits,
            output_mask: ((1u16 << output_bits) - 1) as u8,
            seed: Some(seed),
            cold_rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Reconstruct from a serialised key/value pair (for checkpoint loading).
    #[allow(dead_code)]
    pub fn from_entries(entries: Vec<(u32, u8)>, output_bits: u8, seed: Option<u64>) -> Self {
        let map: HashMap<u32, u8> = entries.into_iter().collect();
        let cold_rng = StdRng::seed_from_u64(seed.unwrap_or(0));
        GeneticCodeDict {
            map,
            output_bits,
            output_mask: ((1u16 << output_bits) - 1) as u8,
            seed,
            cold_rng,
        }
    }
}

impl GeneticCode for GeneticCodeDict {
    fn get(&mut self, key: u32) -> u8 {
        if let Some(&v) = self.map.get(&key) {
            return v;
        }
        // Lazy fill: generate a random output and cache it.
        let v = (self.cold_rng.next_u32() as u8) & self.output_mask;
        self.map.insert(key, v);
        v
    }

    fn set(&mut self, key: u32, value: u8) {
        self.map.insert(key, value);
    }

    fn entries(&self) -> Vec<(u32, u8)> {
        self.map.iter().map(|(&k, &v)| (k, v)).collect()
    }

    /// Crossover algorithm matching Python's `GeneticCodeDict.crossover`:
    ///
    /// 1. Start with a clone of self's map.
    /// 2. For each key in `other`, overlay with 50 % probability.
    /// 3. Apply geometric-gap bit-flip mutation (≈ `mutation_rate * n` flips).
    fn crossover(
        &self,
        other: &dyn GeneticCode,
        mutation_rate: f64,
        rng: &mut dyn RngCore,
    ) -> Box<dyn GeneticCode> {
        let mut child_map = self.map.clone();

        // Overlay entries from other parent with 50 % probability.
        for (k, v) in other.entries() {
            if !child_map.contains_key(&k) || (rng.next_u32() & 1) == 0 {
                child_map.insert(k, v);
            }
        }

        // Geometric-gap bit-flip mutation.
        if mutation_rate > 0.0 && !child_map.is_empty() {
            let keys: Vec<u32> = child_map.keys().cloned().collect();
            let n = keys.len();
            let inv_log = 1.0_f64 / (1.0_f64 - mutation_rate).ln();
            // First skip distance drawn from a geometric distribution.
            let u: f64 = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let mut i = (u.ln() * inv_log) as usize;
            while i < n {
                let k = keys[i];
                let bit = (rng.next_u32() as u8) % self.output_bits;
                *child_map.get_mut(&k).unwrap() ^= 1u8 << bit;
                let u2: f64 = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
                i = i
                    .saturating_add(1)
                    .saturating_add((u2.ln() * inv_log) as usize);
            }
        }

        let child_seed = rng.next_u64();
        Box::new(GeneticCodeDict {
            map: child_map,
            output_bits: self.output_bits,
            output_mask: self.output_mask,
            seed: Some(child_seed),
            cold_rng: StdRng::seed_from_u64(child_seed),
        })
    }

    fn code_type(&self) -> &'static str {
        "GeneticCodeDict"
    }

    fn resp_bits(&self) -> u8 {
        self.output_bits
    }

    fn code_seed(&self) -> Option<u64> {
        self.seed
    }

    fn clone_box(&self) -> Box<dyn GeneticCode> {
        Box::new(GeneticCodeDict {
            map: self.map.clone(),
            output_bits: self.output_bits,
            output_mask: self.output_mask,
            seed: self.seed,
            cold_rng: StdRng::seed_from_u64(self.seed.unwrap_or(0)),
        })
    }
}

// ---------------------------------------------------------------------------
// GeneticCodeList  (secondary — dense Vec for contiguous key spaces)
// ---------------------------------------------------------------------------

/// Dense genetic code backed by a `Vec<u8>`.
///
/// All entries are pre-allocated and initialised with cold-start random
/// values.  Direct `O(1)` index access; no lazy fill needed.  Mirrors
/// Python's `GeneticCodeList`.
#[allow(dead_code)]
pub struct GeneticCodeList {
    code: Vec<u8>,
    output_bits: u8,
    seed: Option<u64>,
}

impl GeneticCodeList {
    /// Create a fully pre-allocated list of `size` entries.
    #[allow(dead_code)]
    pub fn new(size: usize, output_bits: u8, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mask = ((1u16 << output_bits) - 1) as u8;
        let code: Vec<u8> = (0..size).map(|_| (rng.next_u32() as u8) & mask).collect();
        GeneticCodeList {
            code,
            output_bits,
            seed: Some(seed),
        }
    }

    /// Reconstruct from a serialised value list (for checkpoint loading).
    #[allow(dead_code)]
    pub fn from_values(values: Vec<u8>, output_bits: u8, seed: Option<u64>) -> Self {
        GeneticCodeList {
            code: values,
            output_bits,
            seed,
        }
    }
}

impl GeneticCode for GeneticCodeList {
    fn get(&mut self, key: u32) -> u8 {
        self.code[key as usize]
    }

    fn set(&mut self, key: u32, value: u8) {
        self.code[key as usize] = value;
    }

    fn entries(&self) -> Vec<(u32, u8)> {
        self.code
            .iter()
            .cloned()
            .enumerate()
            .map(|(k, v)| (k as u32, v))
            .collect()
    }

    fn crossover(
        &self,
        other: &dyn GeneticCode,
        mutation_rate: f64,
        rng: &mut dyn RngCore,
    ) -> Box<dyn GeneticCode> {
        let other_entries: HashMap<u32, u8> = other.entries().into_iter().collect();
        let n = self.code.len();
        let mut child: Vec<u8> = Vec::with_capacity(n);

        for i in 0..n {
            let self_val = self.code[i];
            let other_val = other_entries.get(&(i as u32)).cloned().unwrap_or(self_val);
            let v = if (rng.next_u32() & 1) == 0 {
                self_val
            } else {
                other_val
            };
            child.push(v);
        }

        // Geometric-gap mutation.
        if mutation_rate > 0.0 && n > 0 {
            let inv_log = 1.0_f64 / (1.0_f64 - mutation_rate).ln();
            let u: f64 = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
            let mut i = (u.ln() * inv_log) as usize;
            while i < n {
                let bit = (rng.next_u32() as u8) % self.output_bits;
                child[i] ^= 1u8 << bit;
                let u2: f64 = (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64);
                i = i
                    .saturating_add(1)
                    .saturating_add((u2.ln() * inv_log) as usize);
            }
        }

        let child_seed = rng.next_u64();
        Box::new(GeneticCodeList {
            code: child,
            output_bits: self.output_bits,
            seed: Some(child_seed),
        })
    }

    fn code_type(&self) -> &'static str {
        "GeneticCodeList"
    }

    fn resp_bits(&self) -> u8 {
        self.output_bits
    }

    fn code_seed(&self) -> Option<u64> {
        self.seed
    }

    fn clone_box(&self) -> Box<dyn GeneticCode> {
        Box::new(GeneticCodeList {
            code: self.code.clone(),
            output_bits: self.output_bits,
            seed: self.seed,
        })
    }
}
