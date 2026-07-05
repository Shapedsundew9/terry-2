# Evolved Mate Preference Trait

## Motivation

### The Blind Crossover Problem

In genetic algorithms operating on finite automata, blind crossover — swapping sub-sequences
of the genome at arbitrary cut points — can produce offspring whose state-transition graphs
contain broken loops: cycles of states that generate no progress. A concrete expression of this
is an automaton that oscillates between two or more states, executing a null action (or any
non-advancing action) indefinitely, yielding zero fitness.

The root cause is that the *meaning* of any state in a finite automaton is distributed across
the entire transition graph. No state is semantically self-contained: what state 3 "does" depends
on what states it can reach, and what those states can reach, forming an interdependent web.
Splicing the first half of one parent's graph onto the second half of another's does not
produce a coherent recombination — it produces a third graph whose connectivity properties
are largely unrelated to either parent.

### Why Selection Alone Does Not Solve It

Zero-fitness automata are quickly eliminated. However, they are continuously recreated
by crossover events between fit parents. As long as two parents carry incompatible partial
structures — for example, one parent turns left at a dead-end, the other turns right, and
the full reversal requires two steps — swapping one mapping out creates a deadlock cycle in
offspring. The population reaches a steady state where a non-trivial fraction of each
generation (~6–18% observed empirically) consists of these broken-loop individuals,
representing a permanent fitness tax on the population.

Adding a null action (a no-op state) to the action vocabulary makes this pathology
more visible but is not the cause: the broken-loop problem exists independently of any
specific action encoding, and removing the null action would not prevent other forms of
the same deadlock.

### The Deeper Issue: Evolvability

The ideal solution is not to patch a specific symptom but to improve the population's
*evolvability* — its capacity to produce viable offspring from crossover. Concretely:
if parents whose genomes are structurally compatible preferentially mate with each other,
the frequency of broken-loop offspring decreases without any domain-specific intervention.

---

## Theory of the Evolved Mate Preference Trait

### Core Idea

Each automaton carries an additional heritable trait — a discrete bit string of fixed length —
called the **compatibility fingerprint**. This fingerprint is not derived from or computed
from the automaton's functional genome; it begins as a random initialisation and acquires
meaning entirely through selection pressure. Mate selection is biased toward partners whose
fingerprint has a low Hamming distance from the selecting automaton's own fingerprint.

Over generations, automata that produce fit offspring when paired with fingerprint-similar
mates will tend to have their fingerprints reinforced toward similarity with successful mates.
Automata that produce unfit offspring will have their fingerprints pushed toward dissimilarity
with unsuccessful mates. The fingerprint therefore converges, population-wide, on a shared
language where *similarity signals compatibility* — not by design but as an emergent property
of selection.

### The Update Rule

The fingerprint is updated within a generation by each surviving parent, once per unique
mating partner, after offspring fitness has been evaluated. The comparison baseline is the
**average fitness of the two parents** (not just the updating parent), to avoid penalising
pairings where one parent is already highly fit.

| Outcome | Action |
|---|---|
| At least one offspring outperforms parent average | Flip one bit of own fingerprint **toward** the mate's fingerprint (reduce Hamming distance by 1) |
| No offspring survive to the next generation | Flip one bit of own fingerprint **away** from the mate's fingerprint (increase Hamming distance by 1) |
| Offspring survive but none exceed parent average | No change |

**Toward** means: choose a random bit position where own fingerprint and mate's fingerprint
differ; flip own bit to match. This reduces the Hamming distance by exactly 1.

**Away** means: choose a random bit position where own fingerprint and mate's fingerprint
agree; flip own bit to mismatch. This increases the Hamming distance by exactly 1.

The single-bit granularity is intentional. It keeps the update signal small, incremental,
and noisy — which is appropriate given that a single offspring cohort is a weak signal
about long-term compatibility. The signal accumulates meaning across many generations.

### Properties of the Rule

- **Symmetric update availability**: both the toward and away operations always have candidate
  bit positions (as long as the fingerprints are not identical or fully complementary), so the
  rule is symmetric in that sense. A negative outcome is just as actionable as a positive one.
- **Approximate commutativity**: if a parent breeds with multiple partners in one generation,
  each partner triggers one bit flip independently. Because each flip targets a randomly
  chosen position from the differing (or matching) positions, the probability of two pairings
  targeting the same bit is low for reasonable fingerprint lengths, and in expectation the
  order of processing does not affect the outcome. The rule can be treated as commutative
  in practice.
- **Dead parents require no update**: an automaton that does not survive to the next
  generation cannot observe offspring fitness and so receives no fingerprint update. This is
  a natural free property of the design.
- **One update per unique partner per generation**: if two automata breed more than once
  within a generation, only one fingerprint update per ordered pair occurs. This prevents
  a single pairing from dominating the fingerprint update and preserves the independence
  of signals from different partners.

### Inheritance

The compatibility fingerprint is inherited via crossover in exactly the same way as the
functional genome — a crossover point is chosen and each offspring receives a prefix from
one parent and a suffix from the other. Because the fingerprint is a flat bit string with
no interdependencies between positions, crossover of the fingerprint does not exhibit the
broken-loop pathology that motivates this design: bit N's meaning does not depend on bit M,
so any combination of two parents' fingerprint prefixes and suffixes produces a valid
fingerprint. This is a structurally important contrast with the functional genome.

### Mutation

The fingerprint has its own independent mutation rate, separate from the genome mutation
rate. This decoupling is deliberate:

- The genome mutation rate must be kept low enough to preserve functional building blocks.
- The fingerprint needs to be *responsive* — it should track changes in population
  compatibility over time and not lag far behind the evolving genome.

The independent mutation rate is a tunable hyperparameter. It should be set higher than the
genome mutation rate in early generations (when the fingerprint signal is uninformative and
exploration is useful) and may be annealed as the population matures.

### Bootstrap Behaviour

In generation zero, all fingerprints are random and carry no compatibility information.
Mate selection in early generations is therefore approximately random, which is correct
behaviour: no signal exists yet. The fingerprint bootstraps its own meaning gradually as
selection accumulates evidence about which pairings produce viable offspring. This requires
no special initialisation or warm-up phase.

---

## Alternatives Considered

### Behavioral Fingerprinting (Phenotypic Compatibility)

Run both candidate parents through a set of standardised probe input sequences and compare
their output trajectories. Compatibility is defined as similarity of response.

*Why not chosen*: requires specifying a canonical probe set, which risks introducing
domain-specific assumptions. The probe set is also static, whereas the evolved fingerprint
adapts with the population.

### Structural Compatibility Signal (Pre-computed Genome Summary)

Compute a summary vector from the genome's structural properties (transition density, state
count, connectivity metrics) and use vector similarity as a mate preference signal.

*Why not chosen*: couples the compatibility signal to the genome structure, which varies by
domain and representation. Also removes the emergent property — the signal would mean
whatever the structural metrics happen to capture, rather than what selection makes it mean.

### Typed Crossover Points

Attach compatibility tags to individual states in the automaton. Crossover is only permitted
at points where tags match, preventing the splicing of structurally incompatible sub-graphs.

*Why not chosen*: addresses crossover at the point of combination rather than at mate
selection, and adds significant complexity to the genome representation. The mate preference
approach acts upstream of crossover, which is conceptually cleaner and more generic.

### Scalar Evolvability Score

A single heritable scalar encoding "my offspring tend to be fit." Acts as a reputation signal
influencing mate selection.

*Why not chosen*: collapses to a proxy for fitness, which is already represented by the
selection mechanism. A scalar carries no directional information about *which* mates are
compatible — only that past offspring were generally good or bad.

---

## Expected Population Dynamics

As the fingerprint signal matures, the population is expected to self-organise into
**compatibility clusters** — groups of automata whose fingerprints are mutually similar and
whose functional genomes are structurally compatible. Mating within a cluster produces lower
rates of broken-loop offspring; mating across clusters produces higher rates, which is
penalised by the update rule.

This is functionally analogous to **reproductive isolation** in biology: populations drift
into compatibility groups without any top-down specification of what compatibility means.
The emergent clustering is a property of the update dynamics, not an engineered outcome.

A potential risk is **premature convergence**: if compatibility clusters form too rapidly,
within-cluster genetic diversity collapses and the population stagnates. The fingerprint
mutation rate is the primary lever for controlling this: higher mutation keeps clusters
porous and diversity high. The independent mutation rate parameter exists precisely to
allow this control without destabilising the functional genome.

---

## Summary of Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Fingerprint representation | Discrete bit string | Simple, generic, no domain knowledge required |
| Compatibility metric | Hamming distance | Computationally cheap, symmetric, well-defined |
| Update granularity | Single bit flip per unique partner | Keeps signal incremental; avoids overreaction to noisy offspring cohorts |
| Fitness comparison baseline | Average of both parents | Avoids penalising fit parents for pairing with less fit mates |
| Fingerprint inheritance | Crossover (same as genome) | Consistent with existing reproduction model; no broken-loop risk for flat bit strings |
| Mutation rate | Independent parameter | Decouples responsiveness of compatibility signal from stability of functional genome |
| Update symmetry | Each parent updates independently | Each parent has its own experience of the pairing; asymmetric updates are natural |
| Domain coupling | None | Fingerprint meaning is entirely emergent; the mechanism is representation-agnostic |
