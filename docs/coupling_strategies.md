# Coupling Strategies for Discrete Automaton Text Prediction

## The Coupling Problem Restated

The current architecture has **partitioned clauses**: clauses `[0..16)` vote on output bit 0, clauses `[16..32)` vote on bit 1, etc. No clause influences more than one output bit. This means:

- To predict `'e'` (0x65 = `01100101`), 8 independent voting systems must coincidentally agree.
- A single beneficial mutation can only affect one output bit at a time.
- Evolution can't discover "when I see `'th'`, predict `'e'`" as a single coherent rule — it must discover 8 separate rules, one per bit, that all happen to trigger on the same input pattern.

The degenerate solution (constant byte) is the only output where 8 independent systems can easily agree, because it requires no input-dependent coordination.

## Strategy 1: Shared Clauses with Response Templates

**Core idea:** Every clause participates in *all* output bits simultaneously. Each clause carries its own "response word" — when it matches, it votes for a complete output pattern.

### Structure

```
Current (partitioned):
  clause[i] = (w_pos, w_neg)           // i ∈ [0, output_bits × num_clauses)
  clause i belongs to output bit (i / num_clauses)

Proposed (shared):
  clause[i] = (w_pos, w_neg, response)  // i ∈ [0, num_clauses)
  response is output_bits wide — which bits this clause votes FOR when matched
```

### Evaluation

```rust
fn evaluate(&self, key: u64) -> u64 {
    let mut votes = vec![0usize; self.output_bits]; // per-bit vote counts
    for i in 0..self.num_clauses {
        if self.w_pos[i] & key == self.w_pos[i] && self.w_neg[i] & key == 0 {
            // This clause matched — cast votes for all bits in its response
            for bit in 0..self.output_bits {
                if self.response[i] & (1 << bit) != 0 {
                    votes[bit] += 1;
                }
            }
        }
    }
    // Threshold per bit (or use total_matching_clauses / 2)
    let mut output = 0u64;
    let total_matching = ...; // count of matching clauses
    for bit in 0..self.output_bits {
        if votes[bit] >= threshold {
            output |= 1 << bit;
        }
    }
    output
}
```

### Why it helps

A single clause can now encode "when I see `'th'`, predict `'e'`" by having:
- `w_pos` matching the bit pattern for `'th'`
- `response = 0x65` (the bits of `'e'`)

A single beneficial mutation that adds or refines one clause immediately affects the full prediction byte. Evolution searches over *coherent byte predictions* rather than independent bit predictions.

### Evolutionary properties

- **Crossover:** Swap whole clauses (w_pos, w_neg, response triple) between parents — same as current per-clause swap but now each clause is a complete "rule."
- **Mutation of response:** Flip bits in the response word. This is a new mutation axis that doesn't exist today. Flipping one response bit changes *which byte* a clause votes for — a small change that explores neighboring characters.
- **Mutation of match:** Same as current w_pos/w_neg mutation. Changes *when* a clause fires but not *what* it predicts.
- **Memory:** `num_clauses` clause triples instead of `output_bits × num_clauses` pairs. For the same total clause count, this is more memory-efficient.

### Threshold question

The threshold becomes trickier. Currently it's `num_clauses/2 + 1` per output bit. With shared clauses, the number of matching clauses varies per input. Options:
- **Fixed threshold** (e.g., half of total clauses) — simple but may be too high/low.
- **Relative threshold** — `votes[bit] > total_matching / 2` — output bit is 1 if more than half of matching clauses want it on. This naturally adapts and means unmatched clauses are "abstaining" rather than voting against.
- **Separate positive/negative response** — each clause has `response_pos` and `response_neg`, voting for and against each bit. Majority among matched clauses wins. Most faithful to Tsetlin semantics.

### Concern

With a fixed threshold: if very few clauses match a given input, the threshold may never be reached → all output bits are 0 → prediction is `'\0'`. This could create a different degenerate attractor (always predict null). The relative threshold avoids this.

---

## Strategy 2: Clause → Codebook Index

**Core idea:** Separate *what* to predict from *when* to predict it. Clauses vote on an index into a codebook of output words. The codebook and clauses co-evolve.

### Structure

```
codebook: [u64; K]                    // K possible output words (e.g., K=32)
clause[i] = (w_pos, w_neg, index)     // index ∈ [0, K)
```

### Evaluation

```rust
fn evaluate(&self, key: u64) -> u64 {
    let mut scores = vec![0usize; K];
    for i in 0..self.num_clauses {
        if self.w_pos[i] & key == self.w_pos[i] && self.w_neg[i] & key == 0 {
            scores[self.index[i]] += 1;
        }
    }
    let winner = scores.iter().enumerate().max_by_key(|(_, &s)| s).unwrap().0;
    self.codebook[winner]
}
```

### Why it helps

- Each codebook entry is a complete output word — coupling is built in.
- Clauses compete to select a codebook entry, not individual bits.
- The codebook can be initialized with the 32 most common bytes in the training data, giving evolution a head start.
- Mutation on the codebook explores "what to predict"; mutation on clause masks explores "when."

### Evolutionary properties

- **Codebook mutation:** Flip bits in codebook entries. Small change = predict a nearby character.
- **Index mutation:** Change which codebook entry a clause points to. A clause that correctly matches `'th'` contexts could switch from voting for codebook entry 5 (space) to entry 12 (`'e'`).
- **Crossover:** Can swap clauses between parents (including their index), and separately crossover codebooks.
- **K parameter:** K=32-64 is likely sufficient. Larger K means more expressive but harder for evolution to converge on winners.

### Concern

Tie-breaking when multiple codebook entries have equal votes. Could use: first-index-wins, random, or secondary scoring.

---

## Strategy 3: Two-Stage Pipeline (Category → Output Table)

**Core idea:** Use Tsetlin clauses to classify the input into one of K categories. A dense lookup table maps each category to a complete output word. This reduces the clause-level problem from 256-class to K-class.

### Structure

```
Stage 1: Tsetlin classifier
  - log2(K) output bits from clauses (partitioned, as current)
  - These bits encode a category 0..K-1

Stage 2: Lookup table
  - table: [u64; K]  — maps category → full output word
```

### Evaluation

```rust
fn evaluate(&self, key: u64) -> u64 {
    // Stage 1: clauses classify input into a category
    let category = self.classify(key);  // existing Tsetlin evaluate, but only log2(K) bits
    // Stage 2: lookup the output for this category
    self.table[category as usize]
}
```

### Why it helps

- The clauses only need to solve a K-class problem (e.g., K=64 → 6 bits). This is dramatically easier than coordinating 16 independent output bits.
- The lookup table provides perfect coupling — each category maps to a complete, coherent output word.
- The table is small enough (64 × 16 bits = 128 bytes) to be fully evolvable.
- The clause problem ("when") and the table problem ("what") are cleanly separated.

### Evolutionary properties

- **Table mutation:** Flip bits in table entries. Very direct — "category 7 should predict `'e'` instead of `' '`."
- **Clause mutation:** Same as current. Refines which inputs map to which category.
- **Crossover:** Can crossover clause sets and tables independently or together.
- **Progressive complexity:** Start with K=8 (3 category bits, 8 possible predictions), scale up as the population improves.

### Concern

K categories might not be enough if different contexts need different predictions for the same category. But K=64 already provides 64 distinct predictions, and the clause system determines which contexts land in which category. The real question is whether the Tsetlin clauses can learn a good K-way partition of the input space.

**Variant:** Make stage 1 produce a *state-dependent* category. Use state_bits to address one of several tables:
```
category = tsetlin_classify(observation)
output = tables[internal_state][category]
```
This gives `2^state_bits × K` distinct output slots, massively expanding capacity.

---

## Strategy 4: Decision List (Priority-Ordered Rules)

**Core idea:** An ordered list of (condition, action) rules. The first rule whose condition matches the input determines the entire output.

### Structure

```
rules: [(w_pos, w_neg, output_word); N]  // ordered by priority
default_output: u64                       // fallback if nothing matches
```

### Evaluation

```rust
fn evaluate(&self, key: u64) -> u64 {
    for rule in &self.rules {
        if rule.w_pos & key == rule.w_pos && rule.w_neg & key == 0 {
            return rule.output_word;
        }
    }
    self.default_output
}
```

### Why it helps

- Each rule produces a complete output word — perfect coupling.
- Rules are interpretable: "if input matches X, output Y."
- A single beneficial mutation (adding one good rule or refining a condition) can immediately improve predictions for specific contexts without disrupting others.
- The default output handles the "common case" while specific rules handle exceptions.

### Evolutionary properties

- **Rule-level crossover:** Inherit rule subsets from each parent, then merge/interleave.
- **Output mutation:** Change what a rule predicts. 
- **Condition mutation:** Change when a rule fires.
- **Order mutation:** Swap rule priorities. This is a unique mutation axis.
- **Rule insertion/deletion:** Add new rules or remove ineffective ones.

### Concern

- Order-dependence makes crossover tricky — rules from parent A may interact badly with rules from parent B at different positions.
- A very specific rule high in the list can shadow broader rules below it.
- Could mitigate by using a "most specific match" criterion instead of priority order (i.e., the matching rule with the most literals set wins).

**Variant: Most-Specific-Match List** — instead of priority order, the matching rule with the highest `(w_pos | w_neg).count_ones()` wins. This eliminates the ordering problem and favors rules that are more specific (more literals constrained), which tends to produce better predictions because they match more precise contexts.

---

## Strategy 5: XOR-Folded Lookup Table

**Core idea:** Compress the 24-bit input space into a manageable table via an evolvable folding function, then use a dense lookup table for the output.

### Structure

```
fold_masks: [u64; table_bits]          // each mask selects + XOR-folds input bits
table: [u64; 2^table_bits]             // dense lookup, each entry is a full output word
```

### Evaluation

```rust
fn evaluate(&self, key: u64) -> u64 {
    let mut index = 0usize;
    for (i, mask) in self.fold_masks.iter().enumerate() {
        // Each table address bit = parity of (input AND mask)
        if (key & mask).count_ones() & 1 == 1 {
            index |= 1 << i;
        }
    }
    self.table[index]
}
```

### Why it helps

- Every table entry is a complete output word — perfect coupling.
- The folding function determines which inputs map to the same table slot. Evolution can learn meaningful groupings (e.g., all "after a vowel" contexts map together).
- With `table_bits = 12`, the table has 4096 entries × 2 bytes = 8KB — small enough to evolve but large enough for meaningful context discrimination.
- XOR-parity folding is a well-studied hash family with good mixing properties.

### Evolutionary properties

- **Table mutation:** Flip bits in table entries — direct, low-risk.
- **Fold mask mutation:** Flip bits in fold masks — changes which inputs get grouped together. This is a higher-level structural mutation.
- **Crossover:** Can crossover fold masks and table entries independently.
- **Gradual refinement:** Start with small `table_bits` (few groups, easy to evolve), increase as needed.

### Concern

- Two very different contexts can collide in the same table slot. The folding function determines collision patterns. Poor fold masks → poor predictions.
- No "don't care" concept — every input maps to exactly one table entry, so a wrong entry hurts all inputs that hash to it.

---

## Comparison Matrix

| Property | Shared Clauses | Codebook | Pipeline | Decision List | XOR-Fold Table |
|---|---|---|---|---|---|
| **Coupling** | Implicit (response word) | Explicit (codebook) | Explicit (table) | Explicit (rule output) | Explicit (table entry) |
| **Minimal mutation for new prediction** | 1 response flip | 1 codebook flip | 1 table flip | 1 rule output flip | 1 table entry flip |
| **Memory** | 3 words/clause | 3 words/clause + K table | clauses + K table | 3 words/rule | fold masks + 2^B table |
| **Capacity** | num_clauses rules | K distinct predictions | K distinct predictions | N rules | 2^B distinct predictions |
| **Crossover naturalness** | High (swap triples) | High (swap clauses + codebook entries) | High (swap independently) | Medium (order-dependent) | High (swap masks + table slices) |
| **Closest to current arch** | ★★★★★ | ★★★★ | ★★★ | ★★ | ★ |
| **Escape from constant prediction** | ★★★★ | ★★★★★ | ★★★★★ | ★★★★★ | ★★★★ |

## Recommendation

**Strategy 1 (Shared Clauses with Response Templates)** is the most natural evolution of the current architecture — same Tsetlin flavor, minimal structural change, and it directly addresses the coupling problem by letting each clause vote on a complete output word. It's also the easiest to retrofit into the existing codebase since you keep the same w_pos/w_neg mask infrastructure and just add a `response: Vec<u64>` per clause.

**Strategy 3 (Two-Stage Pipeline)** is the most *architecturally clean* — it decomposes the problem into "classify context" (hard, handled by clauses) and "map class to byte" (easy, handled by a table). It's also the easiest to reason about and debug.

If picking one to implement first, **Strategy 3** has strong advantages because:
1. The clause system already works for small output widths (it works fine for the maze automata with fewer bits).
2. The table provides immediate, perfect coupling with zero coordination overhead.
3. The two components can be debugged independently.
4. K is an obvious scaling knob.
