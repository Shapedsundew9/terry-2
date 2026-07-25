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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeneticCodeKind {
    Dict,
    List,
    Tsetlin,
}

impl GeneticCodeKind {
    pub fn checkpoint_name(self) -> &'static str {
        match self {
            Self::Dict => "GeneticCodeDict",
            Self::List => "GeneticCodeList",
            Self::Tsetlin => "GeneticCodeTsetlin",
        }
    }
}

#[derive(Clone, Debug)]
pub struct GeneticCodeConfig {
    pub kind: GeneticCodeKind,
    pub tsetlin_clauses: usize,
    pub tsetlin_threshold: Option<f64>,
}

impl Default for GeneticCodeConfig {
    fn default() -> Self {
        Self {
            kind: GeneticCodeKind::Tsetlin,
            tsetlin_clauses: 10,
            tsetlin_threshold: None,
        }
    }
}

pub enum GeneticCode {
    Dict(GeneticCodeDict),
    List(GeneticCodeList),
    Tsetlin(GeneticCodeTsetlin),
}

impl GeneticCode {
    pub fn from_dict_entries(entries: Vec<(u32, u8)>, output_bits: u8, seed: Option<u64>) -> Self {
        Self::Dict(GeneticCodeDict::from_entries(entries, output_bits, seed))
    }

    pub fn from_list_values(values: Vec<u8>, output_bits: u8, seed: Option<u64>) -> Self {
        Self::List(GeneticCodeList::from_values(values, output_bits, seed))
    }

    pub fn new(
        config: &GeneticCodeConfig,
        output_bits: u8,
        input_bits: u8,
        seed: u64,
    ) -> Result<Self, String> {
        match config.kind {
            GeneticCodeKind::Dict => Ok(Self::Dict(GeneticCodeDict::new(output_bits, seed))),
            GeneticCodeKind::List => {
                let size = 1usize
                    .checked_shl(input_bits as u32)
                    .ok_or_else(|| "GeneticCodeList input width is too large".to_string())?;
                Ok(Self::List(GeneticCodeList::new(size, output_bits, seed)))
            }
            GeneticCodeKind::Tsetlin => Ok(Self::Tsetlin(GeneticCodeTsetlin::new(
                output_bits,
                config.tsetlin_clauses,
                input_bits,
                config.tsetlin_threshold,
                seed,
            )?)),
        }
    }

    #[inline]
    pub fn get(&mut self, key: u32) -> u8 {
        match self {
            Self::Dict(code) => code.get(key),
            Self::List(code) => code.get(key),
            Self::Tsetlin(code) => code.evaluate(key as u64),
        }
    }

    pub fn crossover(
        &self,
        other: &Self,
        mutation_rate: f64,
        rng: &mut dyn RngCore,
    ) -> Result<Self, String> {
        if !mutation_rate.is_finite() || !(0.0..=1.0).contains(&mutation_rate) {
            return Err("mutation_rate must be between 0 and 1 inclusive".into());
        }
        match (self, other) {
            (Self::Dict(first), Self::Dict(second)) => Ok(Self::Dict(first.crossover_with_rng(
                second,
                mutation_rate,
                rng,
            ))),
            (Self::List(first), Self::List(second)) => Ok(Self::List(first.crossover_with_rng(
                second,
                mutation_rate,
                rng,
            ))),
            (Self::Tsetlin(first), Self::Tsetlin(second)) => Ok(Self::Tsetlin(
                first.crossover_with_rng(second, mutation_rate, rng)?,
            )),
            _ => Err("genetic-code parents must use the same representation".into()),
        }
    }

    pub fn kind(&self) -> GeneticCodeKind {
        match self {
            Self::Dict(_) => GeneticCodeKind::Dict,
            Self::List(_) => GeneticCodeKind::List,
            Self::Tsetlin(_) => GeneticCodeKind::Tsetlin,
        }
    }

    pub fn code_type(&self) -> &'static str {
        self.kind().checkpoint_name()
    }

    pub fn resp_bits(&self) -> u8 {
        match self {
            Self::Dict(code) => code.output_bits,
            Self::List(code) => code.output_bits,
            Self::Tsetlin(code) => code.output_bits(),
        }
    }

    pub fn code_seed(&self) -> Option<u64> {
        match self {
            Self::Dict(code) => code.seed,
            Self::List(code) => code.seed,
            Self::Tsetlin(code) => code.seed,
        }
    }

    pub fn entries(&self) -> Vec<(u32, u8)> {
        match self {
            Self::Dict(code) => code.entries(),
            Self::List(code) => code.entries(),
            Self::Tsetlin(_) => Vec::new(),
        }
    }

    pub fn as_tsetlin(&self) -> Option<&GeneticCodeTsetlin> {
        match self {
            Self::Tsetlin(code) => Some(code),
            _ => None,
        }
    }
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
#[derive(Clone)]
pub struct GeneticCodeDict {
    map: HashMap<u32, u8>,
    /// Number of bits in the output value (`state_bits + resp_bits`).
    output_bits: u8,
    /// Mask derived from `output_bits`.
    output_mask: u8,
    /// Optional seed stored for checkpoint round-trips.
    seed: Option<u64>,
    /// RNG used for lazy-fill value generation.
    cold_rng: Box<StdRng>,
}

impl GeneticCodeDict {
    /// Create an empty dict pre-seeded with `seed`.
    pub fn new(output_bits: u8, seed: u64) -> Self {
        GeneticCodeDict {
            map: HashMap::new(),
            output_bits,
            output_mask: ((1u16 << output_bits) - 1) as u8,
            seed: Some(seed),
            cold_rng: Box::new(StdRng::seed_from_u64(seed)),
        }
    }

    /// Reconstruct from a serialised key/value pair (for checkpoint loading).
    #[allow(dead_code)]
    pub fn from_entries(entries: Vec<(u32, u8)>, output_bits: u8, seed: Option<u64>) -> Self {
        let map: HashMap<u32, u8> = entries.into_iter().collect();
        let cold_rng = Box::new(StdRng::seed_from_u64(seed.unwrap_or(0)));
        GeneticCodeDict {
            map,
            output_bits,
            output_mask: ((1u16 << output_bits) - 1) as u8,
            seed,
            cold_rng,
        }
    }
}

impl GeneticCodeDict {
    fn get(&mut self, key: u32) -> u8 {
        if let Some(&v) = self.map.get(&key) {
            return v;
        }
        // Lazy fill: generate a random output and cache it.
        let v = (self.cold_rng.next_u32() as u8) & self.output_mask;
        self.map.insert(key, v);
        v
    }

    fn entries(&self) -> Vec<(u32, u8)> {
        self.map.iter().map(|(&k, &v)| (k, v)).collect()
    }

    /// Crossover algorithm matching Python's `GeneticCodeDict.crossover`:
    ///
    /// 1. Start with a clone of self's map.
    /// 2. For each key in `other`, overlay with 50 % probability.
    /// 3. Apply geometric-gap bit-flip mutation (≈ `mutation_rate * n` flips).
    fn crossover_with_rng(&self, other: &Self, mutation_rate: f64, rng: &mut dyn RngCore) -> Self {
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

        let child_seed = rng.next_u32() as u64;
        GeneticCodeDict {
            map: child_map,
            output_bits: self.output_bits,
            output_mask: self.output_mask,
            seed: Some(child_seed),
            cold_rng: Box::new(StdRng::seed_from_u64(child_seed)),
        }
    }
}

// ---------------------------------------------------------------------------
// GeneticCodeTsetlin (flat clause masks, allocation-free lookup)
// ---------------------------------------------------------------------------

/// Clause-voting genetic code equivalent to Python's `GeneticCodeTsetlin`.
///
/// Positive and negative literal masks are stored in row-major order with
/// shape `[output_bits, num_clauses]`. A clause matches when every positive
/// literal is set in the input and every negative literal is clear. Each
/// output bit is enabled when its matching-clause count reaches `threshold`.
#[derive(Clone)]
pub struct GeneticCodeTsetlin {
    pub(crate) w_pos: Vec<u64>,
    pub(crate) w_neg: Vec<u64>,
    output_bits: u8,
    num_clauses: usize,
    input_bits: u8,
    threshold: f64,
    seed: Option<u64>,
}

impl GeneticCodeTsetlin {
    pub fn new(
        output_bits: u8,
        num_clauses: usize,
        input_bits: u8,
        threshold: Option<f64>,
        seed: u64,
    ) -> Result<Self, String> {
        Self::validate_dimensions(output_bits, num_clauses, input_bits)?;
        let threshold = threshold.unwrap_or((num_clauses / 2 + 1) as f64);
        if !threshold.is_finite() {
            return Err("Tsetlin threshold must be finite".into());
        }

        let len = output_bits as usize * num_clauses;
        let mut w_pos = vec![0u64; len];
        let mut w_neg = vec![0u64; len];
        let mut rng = StdRng::seed_from_u64(seed);

        // Match Python's 5% active-literal distribution: 95% ignored,
        // 2.5% required true, and 2.5% required false.
        for bit in 0..input_bits {
            let mask = 1u64 << bit;
            for index in 0..len {
                let sample = unit_f64(&mut rng);
                if sample >= 0.975 {
                    w_neg[index] |= mask;
                } else if sample >= 0.95 {
                    w_pos[index] |= mask;
                }
            }
        }

        Ok(Self {
            w_pos,
            w_neg,
            output_bits,
            num_clauses,
            input_bits,
            threshold,
            seed: Some(seed),
        })
    }

    pub fn from_masks(
        w_pos: Vec<u64>,
        w_neg: Vec<u64>,
        output_bits: u8,
        num_clauses: usize,
        input_bits: u8,
        threshold: f64,
        seed: Option<u64>,
    ) -> Result<Self, String> {
        Self::validate_dimensions(output_bits, num_clauses, input_bits)?;
        let expected = output_bits as usize * num_clauses;
        if w_pos.len() != expected || w_neg.len() != expected {
            return Err(format!(
                "Tsetlin masks must each contain {expected} elements"
            ));
        }
        if !threshold.is_finite() {
            return Err("Tsetlin threshold must be finite".into());
        }
        let input_mask = if input_bits == 64 {
            u64::MAX
        } else {
            (1u64 << input_bits) - 1
        };
        if w_pos
            .iter()
            .chain(&w_neg)
            .any(|mask| mask & !input_mask != 0)
        {
            return Err("Tsetlin mask contains literals outside input_bits".into());
        }
        Ok(Self {
            w_pos,
            w_neg,
            output_bits,
            num_clauses,
            input_bits,
            threshold,
            seed,
        })
    }

    fn validate_dimensions(
        output_bits: u8,
        num_clauses: usize,
        input_bits: u8,
    ) -> Result<(), String> {
        if !(1..=8).contains(&output_bits) {
            return Err("Tsetlin output_bits must be between 1 and 8".into());
        }
        if num_clauses == 0 {
            return Err("Tsetlin num_clauses must be at least 1".into());
        }
        if !(1..=64).contains(&input_bits) {
            return Err("Tsetlin input_bits must be between 1 and 64".into());
        }
        (output_bits as usize)
            .checked_mul(num_clauses)
            .ok_or_else(|| "Tsetlin dimensions are too large".to_string())?;
        Ok(())
    }

    #[inline]
    pub fn evaluate(&self, key: u64) -> u8 {
        let mut output = 0u8;
        for response_bit in 0..self.output_bits as usize {
            let start = response_bit * self.num_clauses;
            let end = start + self.num_clauses;
            let mut votes = 0usize;
            for index in start..end {
                if self.w_pos[index] & key == self.w_pos[index] && self.w_neg[index] & key == 0 {
                    votes += 1;
                }
            }
            if votes as f64 >= self.threshold {
                output |= 1u8 << response_bit;
            }
        }
        output
    }

    pub fn crossover_with_rng(
        &self,
        other: &Self,
        mutation_rate: f64,
        rng: &mut dyn RngCore,
    ) -> Result<Self, String> {
        if self.output_bits != other.output_bits {
            return Err("Tsetlin parents must have the same response bits".into());
        }
        if self.input_bits != other.input_bits {
            return Err("Tsetlin parents must have the same input bits".into());
        }
        if !mutation_rate.is_finite() || !(0.0..=1.0).contains(&mutation_rate) {
            return Err("mutation_rate must be between 0 and 1 inclusive".into());
        }

        let mut child_num_clauses = self.num_clauses;
        if mutation_rate > 0.0 && unit_f64(rng) < mutation_rate {
            if child_num_clauses == 1 || unit_f64(rng) < 0.5 {
                child_num_clauses += 1;
            } else {
                child_num_clauses -= 1;
            }
        }

        let len = self.output_bits as usize * child_num_clauses;
        let mut child_w_pos = Vec::with_capacity(len);
        let mut child_w_neg = Vec::with_capacity(len);

        for response_bit in 0..self.output_bits as usize {
            let self_start = response_bit * self.num_clauses;
            let other_start = response_bit * other.num_clauses;
            for _ in 0..child_num_clauses {
                let (source, source_start, source_len) = if unit_f64(rng) < 0.5 {
                    (self, self_start, self.num_clauses)
                } else {
                    (other, other_start, other.num_clauses)
                };
                let source_index = source_start + random_index(rng, source_len);
                child_w_pos.push(source.w_pos[source_index]);
                child_w_neg.push(source.w_neg[source_index]);
            }
        }

        if mutation_rate > 0.0 {
            for index in 0..len {
                if unit_f64(rng) >= mutation_rate {
                    continue;
                }
                let bit = random_index(rng, self.input_bits as usize) as u8;
                let mask = 1u64 << bit;
                let was_positive = child_w_pos[index] & mask != 0;
                let was_negative = child_w_neg[index] & mask != 0;
                let alternative = (rng.next_u32() & 1) as u8;

                child_w_pos[index] &= !mask;
                child_w_neg[index] &= !mask;
                match (was_positive, was_negative, alternative) {
                    (false, false, 0) => child_w_pos[index] |= mask,
                    (false, false, 1) => child_w_neg[index] |= mask,
                    (true, false, 1) => child_w_neg[index] |= mask,
                    (false, true, 1) => child_w_pos[index] |= mask,
                    _ => {}
                }
            }
        }

        let threshold = if child_num_clauses == self.num_clauses {
            self.threshold
        } else {
            self.threshold * child_num_clauses as f64 / self.num_clauses as f64
        };

        Self::from_masks(
            child_w_pos,
            child_w_neg,
            self.output_bits,
            child_num_clauses,
            self.input_bits,
            threshold,
            Some(rng.next_u32() as u64),
        )
    }

    pub fn output_bits(&self) -> u8 {
        self.output_bits
    }

    pub fn num_clauses(&self) -> usize {
        self.num_clauses
    }

    pub fn input_bits(&self) -> u8 {
        self.input_bits
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    pub fn positive_masks(&self) -> &[u64] {
        &self.w_pos
    }

    pub fn negative_masks(&self) -> &[u64] {
        &self.w_neg
    }
}

#[inline]
fn unit_f64(rng: &mut dyn RngCore) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

#[inline]
fn random_index(rng: &mut dyn RngCore, upper: usize) -> usize {
    (rng.next_u64() as usize) % upper
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
#[derive(Clone)]
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

impl GeneticCodeList {
    fn get(&mut self, key: u32) -> u8 {
        self.code[key as usize]
    }

    fn entries(&self) -> Vec<(u32, u8)> {
        self.code
            .iter()
            .cloned()
            .enumerate()
            .map(|(k, v)| (k as u32, v))
            .collect()
    }

    fn crossover_with_rng(&self, other: &Self, mutation_rate: f64, rng: &mut dyn RngCore) -> Self {
        let n = self.code.len();
        let mut child: Vec<u8> = Vec::with_capacity(n);

        for i in 0..n {
            let self_val = self.code[i];
            let other_val = other.code.get(i).copied().unwrap_or(self_val);
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

        let child_seed = rng.next_u32() as u64;
        GeneticCodeList {
            code: child,
            output_bits: self.output_bits,
            seed: Some(child_seed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tsetlin_evaluates_known_clause_masks() {
        let code = GeneticCodeTsetlin::from_masks(
            vec![0b0001, 0b0001, 0b0010, 0],
            vec![0, 0b0010, 0, 0b0001],
            2,
            2,
            4,
            2.0,
            Some(7),
        )
        .unwrap();

        assert_eq!(code.evaluate(0b0001), 0b01);
        assert_eq!(code.evaluate(0b0010), 0b10);
    }

    #[test]
    fn tsetlin_initialization_is_seeded_and_valid() {
        let first = GeneticCodeTsetlin::new(6, 10, 13, None, 42).unwrap();
        let second = GeneticCodeTsetlin::new(6, 10, 13, None, 42).unwrap();

        assert_eq!(first.positive_masks(), second.positive_masks());
        assert_eq!(first.negative_masks(), second.negative_masks());
        assert_eq!(first.threshold(), 6.0);
        assert!(first
            .positive_masks()
            .iter()
            .zip(first.negative_masks())
            .all(|(positive, negative)| positive & negative == 0));
    }

    #[test]
    fn tsetlin_crossover_inherits_complete_clause_pairs() {
        let parent_a = GeneticCodeTsetlin::from_masks(
            vec![1, 2, 4, 8, 16, 32],
            vec![64, 128, 3, 5, 6, 9],
            2,
            3,
            8,
            2.0,
            Some(1),
        )
        .unwrap();
        let parent_b = GeneticCodeTsetlin::from_masks(
            vec![10, 20, 40, 80],
            vec![11, 21, 41, 81],
            2,
            2,
            8,
            2.0,
            Some(2),
        )
        .unwrap();
        let mut rng = StdRng::seed_from_u64(9);

        let child = parent_a
            .crossover_with_rng(&parent_b, 0.0, &mut rng)
            .unwrap();

        assert_eq!(child.num_clauses(), parent_a.num_clauses());
        for response_bit in 0..child.output_bits() as usize {
            let mut parent_pairs = Vec::new();
            for parent in [&parent_a, &parent_b] {
                let start = response_bit * parent.num_clauses();
                for index in start..start + parent.num_clauses() {
                    parent_pairs.push((parent.w_pos[index], parent.w_neg[index]));
                }
            }
            let child_start = response_bit * child.num_clauses();
            for index in child_start..child_start + child.num_clauses() {
                assert!(parent_pairs.contains(&(child.w_pos[index], child.w_neg[index])));
            }
        }
    }

    #[test]
    fn tsetlin_forced_mutation_changes_structure_and_valid_literals() {
        let parent = GeneticCodeTsetlin::from_masks(
            vec![0, 0, 0, 0, 0, 0],
            vec![0, 0, 0, 0, 0, 0],
            2,
            3,
            8,
            2.0,
            Some(1),
        )
        .unwrap();
        let other = GeneticCodeTsetlin::from_masks(
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 0],
            2,
            2,
            8,
            2.0,
            Some(2),
        )
        .unwrap();
        let mut rng = StdRng::seed_from_u64(4);

        let child = parent.crossover_with_rng(&other, 1.0, &mut rng).unwrap();

        assert_eq!(child.num_clauses().abs_diff(parent.num_clauses()), 1);
        assert_eq!(
            child.threshold(),
            parent.threshold() * child.num_clauses() as f64 / parent.num_clauses() as f64
        );
        for (positive, negative) in child.w_pos.iter().zip(&child.w_neg) {
            assert_eq!(positive & negative, 0);
            let literals = positive | negative;
            assert_eq!(literals.count_ones(), 1);
        }
    }

    #[test]
    fn tsetlin_rejects_invalid_mutation_rate() {
        let parent = GeneticCodeTsetlin::new(2, 2, 8, None, 1).unwrap();
        let mut rng = StdRng::seed_from_u64(2);

        for rate in [-0.01, 1.01, f64::NAN] {
            assert!(parent.crossover_with_rng(&parent, rate, &mut rng).is_err());
        }
    }
}
