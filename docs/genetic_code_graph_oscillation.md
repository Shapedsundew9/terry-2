# Genetic code oscillation diagnosis

## Symptom

`MazeAutomaton` driven by the older graph-based genetic code characteristically
runs up and down a single corridor: it reverses (two turns to go back on itself)
but rarely turns a single corner to continue along a perpendicular axis. Fitness
shows a short burst of improvement and then stalls flat. The same automaton
driven by `GeneticCodeDict` navigates corners and reaches higher fitness.

## Investigation

All numbers below come from throwaway probes run against the maze parameters
(`env_bits=9`, `state_bits=4`, `resp_bits=2`, so the graph has `input_bits=13`,
`resp_bits=6`, `num_nodes=64`). The action is the top 2 bits of the 6-bit output
(`0=forward`, `1=left`, `2=right`, `3=invalid/no-op`); the new internal state is
the low 4 bits.

### 1. The behaviour is real and graph-specific

Evolving champions (100 ticks/gen, 60 gens, pop 100) and replaying them:

| representation | best fitness | single-turn corners | double-turn reversals |
| --- | --- | --- | --- |
| graph-based variant | ~25 | ~5 (as low as 0) | ~11 (as high as 15) |
| `GeneticCodeDict`  | higher, mean ~6 | ~7 | ~10 |

The graph reverses far more than it corners; the dict is balanced. So
"turns twice, never once" is a genuine graph-specific signature.

### 2. It is NOT a bit-level encoding collapse

Per-output-bit entropy over all 2^13 inputs (averaged over random graphs):

- state bits 0-3: ~0.77-0.84 (of max 1.0)
- action bits 4-5: ~0.78-0.80
- `P(action bit4 == bit5) = 0.49` (independent)

So the action signal is not being squashed, and the two turn codes are not
structurally suppressed by the `state_bits .. state_bits+resp_bits` slicing.

### 3. Root cause: `crossover` does not preserve building blocks

The older graph-based crossover was index-aligned (child node `p` was taken whole
from parent A or B). But the genome is **position-dependent**: node `p`'s
semantics depend on the values computed by *all* earlier nodes, because `in_a` /
`in_b` are absolute value-array indices. Splicing parent A's node onto a prefix
that is a mix of A and B nodes computes something unrelated to what it meant in A.

Measured on the (state,env) -> action table (mutation disabled):

- two random parents disagree on **74%** of action entries
- a child disagrees with its **closest** parent on **71%**

If crossover preserved building blocks, child-vs-closest would be far smaller
than parent-vs-parent. It is not — **crossover behaves like a random re-draw, not
recombination.** This is the mechanistic explanation for "burst then flat stall":
the initial population contains a few lucky individuals (the burst), selection
concentrates them, then crossover scrambles every partial improvement, so nothing
better than the easy attractor ever stabilises.

`GeneticCodeDict` does not have this problem because its crossover is *key*-aligned
and each key is independent, so inherited entries are genuine building blocks.

### 4. Why reversals specifically (geometry + coupling)

- Corridors are 1-wide, so a single crude "wall ahead => turn" reflex
  **automatically produces a reversal**: turn once, the rotated 3x3 view still
  shows wall-ahead (the corridor wall), so the same reflex fires again, now clear,
  go forward. Reversal reuses one reflex twice.
- Cornering needs a finer, env-conditional response (distinguish "wall ahead +
  side open" from "wall ahead + sides closed") plus a turn-then-forward pair
  across two different views — a more complex sub-program that destructive
  crossover keeps wrecking and that the dict can encode per-key for free.
- Secondary factor: the action bits and next-state bits are XOR-folded from the
  **same** trailing nodes, so the policy and the memory dynamics are entangled —
  you cannot tune one without perturbing the other, unlike the dict's separate
  fields.

## Suggested fixes (ranked)

1. **Mutation-driven reproduction (root-cause fix).** For position-dependent
   Cartesian / linear-GP genomes, recombination is destructive; the standard
   practice is to reproduce by cloning one parent and applying per-gene mutation.
   Make crossover clone-one-parent + tunable per-gene mutation, keeping true
   recombination optional.
2. **Decouple action from state.** Fold a separate node group for the action bits
   versus the state bits so the policy and the memory dynamics can evolve
   independently.
3. **Tighten the action encoding.** Code `3` currently wastes 25% of outputs on a
   no-op "invalid" and makes "forward" the unique `00` code; mapping `3` to
   forward (or using mod-3) removes the slight anti-forward bias.

## Caveat

The obvious fix #1 was spot-tested (clone + per-gene mutation, no crossover).
It makes champions corner more on some seeds, but **averaged over 6 seeds the
fitness gain was within noise** (best ~28 vs ~25.5). Mutation-only alone is not a
silver bullet in short runs; the action/state entanglement (#2) and the geometry
trap also contribute. A proper multi-seed benchmark is needed to quantify any fix.
