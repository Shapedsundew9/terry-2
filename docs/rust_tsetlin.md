# Rust Tsetlin Maze Runner

The Rust `maze-runner` uses `GeneticCodeTsetlin` by default. Dict and dense List
codes remain available for comparison.

## Fresh experiments

```bash
cargo run --release --bin maze-runner -- \
  --name baseline-tsetlin \
  --generations 10000 \
  --populations 100 \
  --parallel 12
```

Select a model explicitly with `--code-type tsetlin|dict|list`:

```bash
cargo run --release --bin maze-runner -- \
  --name tsetlin-20-clauses \
  --code-type tsetlin \
  --tsetlin-clauses 20

cargo run --release --bin maze-runner -- \
  --name dict-control \
  --code-type dict
```

The Tsetlin threshold is always a strict majority of the configured clause
count. Clause count and threshold remain fixed during crossover. Input width is
`9 + state_bits`; output width is `state_bits + 2`. Tsetlin starts with 4 clauses
and a 5% active-literal probability, matching the Python
`GeneticCodeTsetlin` defaults.

Selection fingerprints are optional for fresh Rust runs:

```bash
cargo run --release --bin maze-runner -- \
  --name fingerprint-tsetlin \
  --fingerprint \
  --fingerprint-bits 4 \
  --fingerprint-tournament-k 4 \
  --fingerprint-mutation-rate 0.01
```

## Resume

`--resume` continues one population. `--generations` is the target total, not an
additional count. For example, this runs 200 generations from a generation-800
checkpoint:

```bash
cargo run --release --bin maze-runner -- \
  --name resumed-gen-1000 \
  --resume runs/2026-07-25T20-01-25-932326/gen_000800 \
  --generations 1000 \
  --maze-seed 42 \
  --side-length-bits 6
```

The `.toml` or `.npz` suffix may be supplied; the shared stem is used. The
checkpoint supplies population size, state width, genetic-code representation,
fixed clause count, fingerprint configuration, and existing fitness history.
CLI values still control ticks, restarts, mutation rate,
checkpoint interval, continuation seed, output directory, and target generation.

The checkpoint environment name is used automatically. `--maze-name` can
override it when deliberately loading against a differently named Maze.

Schema-v1 checkpoints do not contain the maze wall/goal arrays or RNG states.
The runner therefore reconstructs the external Maze from `--maze-seed` and
`--side-length-bits`, then deterministically resets episode-local automaton state
before continuing. Python and Rust do not promise the same maze or random stream
for the same numeric seed. Rust does promise repeatable continuation for the
same Rust checkpoint, configuration, and seed.

## Checkpoint compatibility

Rust writes the existing schema-v1 TOML/NPZ format:

- Dict: `automaton_N_keys` and `automaton_N_values` as `int64` arrays.
- List: `automaton_N_values` as an `int64` array.
- Tsetlin: `automaton_N_w_pos` and `automaton_N_w_neg` as row-major `uint64`
  matrices shaped `[response_bits, clauses]`.
- Energy grids are `uint8`; full fitness history is `float64`.
- Tsetlin metadata includes response width, clause count, input width, derived
  threshold, and seed. Legacy threshold values are accepted but the strict
  majority is derived from the clause count when loading.
- Fingerprint configuration and per-automaton fingerprint values round-trip.

Python `Population.load` can load Rust checkpoints, and Rust can load Python
Dict, List, and Tsetlin checkpoints. Previous-generation fingerprint pairing
references are not serialized by Python, so the first feedback update after any
resume is skipped in both implementations.

## PostgreSQL

No database migration is required. The existing `experiments` and
`generation_stats` schemas and upsert behavior are unchanged. Model,
fingerprint, and resume-source details are stored in `experiments.params_json`.
A resumed run is a new named experiment with one population and complete
history from generation 1 through the target.

## Performance benchmark

```bash
cargo bench --bench tsetlin
```

The benchmark measures the default 6-output-bit by 4-clause lookup and
crossover paths. Lookup uses flat row-major `u64` masks and performs no heap
allocation or virtual dispatch per tick.
