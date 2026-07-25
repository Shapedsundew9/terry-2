use rand::RngCore;

#[derive(Clone, Debug)]
pub struct FingerprintConfig {
    pub bits: u8,
    pub tournament_k: usize,
    pub mutation_rate: f64,
}

impl FingerprintConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=64).contains(&self.bits) {
            return Err("fingerprint bits must be between 1 and 64".into());
        }
        if self.tournament_k == 0 {
            return Err("fingerprint tournament_k must be at least 1".into());
        }
        if !self.mutation_rate.is_finite() || !(0.0..=1.0).contains(&self.mutation_rate) {
            return Err("fingerprint mutation_rate must be between 0 and 1".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct SelectionFingerprint {
    bits: u8,
    value: u64,
}

impl SelectionFingerprint {
    pub fn random(bits: u8, rng: &mut dyn RngCore) -> Self {
        Self::with_value(bits, rng.next_u64())
    }

    pub fn with_value(bits: u8, value: u64) -> Self {
        let mask = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        Self {
            bits,
            value: value & mask,
        }
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    pub fn value(&self) -> u64 {
        self.value
    }

    pub fn hamming(&self, other: &Self) -> u32 {
        (self.value ^ other.value).count_ones()
    }

    pub fn crossover(&self, other: &Self, rng: &mut dyn RngCore) -> Self {
        let selector = rng.next_u64();
        Self::with_value(
            self.bits,
            (self.value & selector) | (other.value & !selector),
        )
    }

    pub fn mutate(&mut self, mutation_rate: f64, rng: &mut dyn RngCore) {
        if mutation_rate > 0.0 && unit_f64(rng) < mutation_rate {
            self.value ^= 1u64 << random_index(rng, self.bits as usize);
        }
    }

    pub fn flip_toward(&mut self, other: &Self, rng: &mut dyn RngCore) {
        if let Some(bit) = random_set_bit(self.value ^ other.value, rng) {
            self.value ^= 1u64 << bit;
        }
    }

    pub fn flip_away(&mut self, other: &Self, rng: &mut dyn RngCore) {
        let mask = if self.bits == 64 {
            u64::MAX
        } else {
            (1u64 << self.bits) - 1
        };
        if let Some(bit) = random_set_bit(!(self.value ^ other.value) & mask, rng) {
            self.value ^= 1u64 << bit;
        }
    }
}

fn random_set_bit(mask: u64, rng: &mut dyn RngCore) -> Option<u32> {
    let count = mask.count_ones();
    if count == 0 {
        return None;
    }
    let mut target = (rng.next_u32() % count) as usize;
    for bit in 0..64 {
        if mask & (1u64 << bit) != 0 {
            if target == 0 {
                return Some(bit);
            }
            target -= 1;
        }
    }
    None
}

#[inline]
fn unit_f64(rng: &mut dyn RngCore) -> f64 {
    (rng.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

#[inline]
fn random_index(rng: &mut dyn RngCore, upper: usize) -> usize {
    (rng.next_u64() as usize) % upper
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn toward_and_away_change_hamming_by_one() {
        let mut rng = StdRng::seed_from_u64(1);
        let other = SelectionFingerprint::with_value(8, 0b0000_1111);
        let mut toward = SelectionFingerprint::with_value(8, 0b1111_1111);
        let before = toward.hamming(&other);
        toward.flip_toward(&other, &mut rng);
        assert_eq!(toward.hamming(&other), before - 1);

        let mut away = other.clone();
        away.flip_away(&other, &mut rng);
        assert_eq!(away.hamming(&other), 1);
    }

    #[test]
    fn forced_mutation_flips_one_bit() {
        let mut rng = StdRng::seed_from_u64(2);
        let mut fingerprint = SelectionFingerprint::with_value(8, 0b1010_1010);
        let original = fingerprint.value();
        fingerprint.mutate(1.0, &mut rng);
        assert_eq!((original ^ fingerprint.value()).count_ones(), 1);
    }
}
