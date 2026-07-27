# Rust WikiText Runner

The `wiki-runner` binary evolves byte-level next-character predictors against
WikiText-2. It uses the same PostgreSQL experiment store, Rayon population pool,
selection fingerprints, and schema-v1 TOML/NPZ checkpoints as `maze-runner`.

## Run WikiText-2

```bash
cargo run --release --bin wiki-runner -- \
  --name wikitext2-baseline-state8-tsetlin
```

The defaults match `arc3_agi/wiki_runner.py`: 100 populations, 12 workers,
1,000 generations, 1,000 ticks per generation, one restart, 100 automata per
population, 8 state bits, 8 prediction bits, 16 Tsetlin clauses, and selection
fingerprints enabled with 4 bits and a tournament size of 4.

Use smaller values for a smoke run:

```bash
cargo run --release --bin wiki-runner -- \
  --name wiki-smoke \
  --populations 1 \
  --parallel 1 \
  --generations 1 \
  --ticks 10 \
  --pop-size 4
```

`--database-url` overrides `DATABASE_URL`. Otherwise the runner uses the same
local PostgreSQL default as `maze-runner`. Experiment names are unique: a
completed name is skipped, a failed name is reclaimed, and an active name is
rejected.

## Dataset And Cache

By default the runner downloads the single Parquet split from
`Salesforce/wikitext`, config `wikitext-2-raw-v1`, split `train`. It validates
the temporary Parquet file before atomically moving it into:

```text
${XDG_CACHE_HOME:-$HOME/.cache}/terry-2/wikitext/wikitext-2-raw-v1/train.parquet
```

Change the split or cache root with `--dataset-split` and `--cache-dir`. For
offline runs, provide the Hugging Face Parquet split directly:

```bash
cargo run --release --bin wiki-runner -- \
  --name wiki-local \
  --dataset-path /data/train-00000-of-00001.parquet
```

An explicit path bypasses URL and cache resolution. The Parquet file must have
a non-null UTF-8 `text` column with at least one non-empty value. Empty strings
are filtered exactly as in Python; all other UTF-8 bytes, including trailing
newlines, are preserved.

## Prediction Semantics

Each observation packs the previous and current bytes into 16 bits. At the
first byte of a text, the observation contains only the current byte. The low
8 output bits become the next internal state and the high 8 bits predict the
next byte. The target after the final byte is the zero sentinel. Fitness is the
fraction of correct predictions during the restart.

Wiki runs intentionally expose only `GeneticCodeTsetlin`. A dense List code
would allocate approximately 32 MiB per automaton for the 24-bit input space.
Maze Dict and List support is unchanged.

## Checkpoint And Resume

The default writes a checkpoint at generation 1,000. Set another cadence with
`--checkpoint-interval`; zero writes only at the target generation.

Resume one population by supplying either sidecar path or its shared stem:

```bash
cargo run --release --bin wiki-runner -- \
  --name wiki-resumed-2000 \
  --resume runs/RUN_ID/pop_0/gen_001000 \
  --generations 2000
```

`--generations` is the target total. The checkpoint supplies population size,
state width, Tsetlin masks, fingerprints, and prior fitness history. The CLI
still supplies the external dataset, ticks, restarts, mutation rate,
continuation seed, checkpoint cadence, and output location.

Rust writes Python-compatible `ByteEnv`/`WikiAutomaton` schema-v1 checkpoints,
and each implementation can load checkpoints written by the other. Dataset
bytes and random-generator state are not embedded, so resume requires the same
logical dataset and does not promise identical Python and Rust random streams.
Rust continuation is repeatable for the same checkpoint, dataset, settings,
and seed.

## Benchmark

```bash
cargo bench --bench tsetlin
```

The benchmark includes both the Maze `6 x 4` and Wiki `16 x 16` Tsetlin lookup
and crossover shapes.
