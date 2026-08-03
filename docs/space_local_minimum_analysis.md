# WikiText Tsetlin Automata: Space Local Minimum Analysis

## The Problem

Your population has converged to fitness ~0.224 by generation 40, with the historical best of 0.266 at generation 15 — and it's now *regressing*. The `P` row shows `~` for every byte: the automaton is predicting the same character everywhere. That character is almost certainly `0x20` (space), which appears ~18-22% of the time in English prose, perfectly matching the observed fitness range.

## Why This Happens: Root Cause Analysis

### 1. Fitness Landscape Topology

The fitness function `right / total` is **raw accuracy**. For a 256-class prediction problem, the optimal degenerate strategy is always: **predict the mode of the marginal distribution**. Space is that mode. This creates a massive, flat basin of attraction — any automaton that stumbles into "always predict space" immediately outperforms random guessing (~0.4% accuracy for uniform random) by ~50×.

### 2. The Tsetlin Architecture Makes It Worse

Looking at [evaluate()](file:///workspaces/terry-2/src/genetic_code.rs#L410-L426):

```rust
if votes >= self.threshold {  // threshold = num_clauses/2 + 1 = 9
    output |= 1u64 << response_bit;
}
```

Each output bit is decided by a majority vote of 16 clauses with threshold 9. At initialization (95% don't-care, 2.5% pos, 2.5% neg), most clauses match *everything* because they have almost no literals. So the initial output bits are essentially determined by whether >8 of 16 near-empty clauses happen to match — which is roughly a coin flip per bit, biased heavily toward "all match → bit=1".

The key issue: **output bits are independent**. There's no structural coupling that would force the 8 prediction bits to form a coherent byte. Evolution finds it far easier to lock 8 independent bits to match the most common byte than to learn context-dependent conditionals.

### 3. Mutation Can't Escape

The [Tsetlin mutation](file:///workspaces/terry-2/src/genetic_code.rs#L525-L553) operates per-clause:
- Active bits → don't-care
- Don't-care bits → randomly active

With `mutation_rate_exponent = 7` (rate ≈ 0.78% per literal per clause), this is a gentle perturbation. Once the population converges to "predict space," a single mutation affecting one clause for one output bit rarely improves fitness. The mutation would need to simultaneously:
1. Make a clause match *specific* contexts (not all inputs)
2. Affect the right output bit to change the prediction for that context
3. Get the prediction *right* for that context more often than it gets it wrong for others

The probability of a beneficial mutation is vanishingly small because changing one clause changes the prediction for *all* inputs where that clause is pivotal.

### 4. Selection Pressure Collapses Diversity

The fitness history confirms this:
| Gen | Min | Max | Mean | Spread |
|-----|-----|-----|------|--------|
| 1   | 0.008 | 0.184 | 0.069 | 0.176 |
| 15  | 0.146 | **0.266** | 0.204 | 0.120 |
| 40  | 0.182 | 0.224 | 0.212 | **0.042** |

The min-max spread collapsed from 0.176 to 0.042 — the population is genetically uniform. The fingerprint-based mate selection tries to maintain diversity, but with only 4 bits and the entire population converged to the same phenotype, there's no meaningful diversity signal.

---

## Strategies to Escape

### Tier 1: Changes to the Fitness Function (Highest Impact)

#### Strategy A: Character-Class Balanced Accuracy

Instead of raw accuracy, weight each correct prediction inversely by its character's frequency. This removes the reward for "always predict the mode."

```rust
// In WikiAutomaton::tick(), replace the simple right/total with:
// Pre-compute per-byte weights: weight[b] = 1.0 / frequency[b] 
// Then: self.weighted_right += weight[actual] when prediction == actual
// fitness = weighted_right / total
```

This makes correctly predicting a rare letter like 'q' worth ~100× more than predicting space, eliminating the "predict-the-mode" basin.

#### Strategy B: Surprisal-Based (Log-Loss Proxy) Scoring

Award partial credit for "close" predictions at the bit level:

```rust
// Count matching bits between prediction and actual
let matching_bits = 8 - (prediction ^ actual).count_ones();
self.score += matching_bits as f64 / 8.0;
```

This creates a much smoother fitness landscape — an automaton that predicts 'e' when the answer is 'f' (differing by one bit: 0x65 vs 0x66) gets 7/8 credit instead of 0.

#### Strategy C: Exclude Spaces from Scoring

The simplest surgical fix — don't award fitness for correctly predicting spaces:

```rust
if prediction == actual && actual != b' ' {
    self.right += 1;
}
// or keep a separate space_right counter and compute fitness from non-space accuracy
```

### Tier 2: Changes to the Evolutionary Dynamics (Medium Impact)

#### Strategy D: Fitness Sharing / Niching

Divide the population into niches that must predict *different* character classes well. Automata that share the same prediction strategy compete only within their niche:

```rust
// Compute a "behavior fingerprint" based on what the automaton actually predicts
// (not just accuracy), then apply fitness sharing: 
// adjusted_fitness = fitness / count_of_similar_behaviors
```

#### Strategy E: Island Model with Migration

Instead of one population of 100, run separate islands with different conditions (different text subsets, different scoring) and periodically exchange top individuals. You already have `total_populations = 100` — make them interact:

```rust
// Every N generations, migrate top K automata between adjacent populations
// Different populations could use different observation_bytes or fitness variants
```

#### Strategy F: Novelty Search Component

Add a novelty bonus to fitness based on how different an automaton's prediction *pattern* is from the population average:

```rust
// Record the prediction string for each automaton over an evaluation window
// fitness = α * accuracy + (1-α) * novelty_score
```

#### Strategy G: Adaptive Mutation Rate (Self-Adaptive Evolution Strategy)

When fitness stagnates for N generations, automatically increase mutation rate:

```rust
if last_K_generations_max_fitness_unchanged() {
    mutation_rate_exponent = max(mutation_rate_exponent - 2, 1); // Higher mutation
}
```

### Tier 3: Changes to the Tsetlin Architecture (Deeper Changes)

#### Strategy H: Conditional Output Coupling

The fundamental issue is that 8 prediction bits are decided independently. Add clause-level coupling:

```rust
// Instead of independent per-bit voting, have clauses that vote on 
// output *byte values* directly. Each clause proposes a specific byte,
// and the prediction is the byte with the most votes.
```

This is architecturally different but would let a single clause say "when I see 'th', predict 'e'" rather than trying to coordinate 8 independent bit decisions.

#### Strategy I: More Clauses + Lower Threshold

With only 16 clauses per output bit and threshold 9, the system has very low capacity. A single clause flipping changes the output for all inputs it matches. More clauses = finer-grained decisions:

```
--tsetlin-clauses 64     # 4× more clauses
--mutation-rate 5         # higher mutation to explore faster
```

#### Strategy J: Reduce Observation Window

With `observation_bytes = 2` → 16 observation bits + 8 state bits = 24 input bits, the Tsetlin clause input space is `2^24 ≈ 16M` states. With only 256 clauses total, each clause covers a huge input region. Reducing to `observation_bytes = 1` (8 input bits) → `2^16 = 65K` states might allow more targeted clauses.

```
--observation-bytes 1 --tsetlin-clauses 32
```

### Tier 4: Quick Experiments to Try Now

#### Experiment 1: Higher Mutation + More Restarts
```bash
cargo run --release --bin wiki_runner -- \
  --name "wiki-high-mutation" \
  --mutation-rate 3 \
  --restarts 5 \
  --ticks 200 \
  --generations 200 \
  --populations 20 \
  --parallel 12
```
Much higher mutation rate (1/8 vs 1/128) to break out of convergence. More restarts per generation for more robust fitness estimates.

#### Experiment 2: Bitwise Partial Credit (requires code change)
Modify [`WikiAutomaton::tick()`](file:///workspaces/terry-2/src/wiki.rs#L135-L161) to use bit-matching fitness — this is the single highest-leverage code change.

#### Experiment 3: Larger Capacity
```bash
cargo run --release --bin wiki_runner -- \
  --name "wiki-big-tsetlin" \
  --tsetlin-clauses 64 \
  --state-bits 4 \
  --observation-bytes 1 \
  --mutation-rate 5 \
  --pop-size 200 \
  --generations 500 \
  --populations 10 \
  --parallel 10
```
Fewer input bits but more clauses per bit, giving better coverage of the (smaller) input space.

---

## Recommended Priority Order

> [!IMPORTANT]
> The single most impactful change is **Strategy B (bitwise partial credit)**. It simultaneously:
> 1. Smooths the fitness landscape (gradient signal for "almost right" predictions)
> 2. Destroys the "predict-the-mode" basin (getting 7/8 bits right on a common char is worth less than 8/8 on a rare char, since rarer chars differ in more bits)  
> 3. Requires minimal code changes (3 lines in `tick()`)

After that:
1. **Strategy C** (exclude spaces) as a quick experiment to validate the diagnosis
2. **Strategy G** (adaptive mutation) as low-hanging infrastructure
3. **Experiment 3** parameters (smaller input space + more clauses)
4. **Strategy D** (fitness sharing) for sustained diversity
