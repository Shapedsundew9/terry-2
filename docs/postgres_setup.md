# PostgreSQL Experiment Store

`arc3_agi.experiment.ExperimentStore` uses PostgreSQL for experiment metadata
and generation statistics. Multiple machines can point at the same PostgreSQL
server and safely coordinate experiment names because `run_experiment` claims a
unique experiment row before any population workers start.

## Connection Configuration

The store resolves its database URL in this order:

1. An explicit `database_url` argument passed to `ExperimentStore` or
   `maze_runner.run_experiment`.
2. The `DATABASE_URL` environment variable.
3. The local development default:

```bash
postgresql://arc3_agi:arc3_agi@localhost:5432/arc3_agi
```

For tests, `arc3_agi/tests/test_experiment.py` uses `TEST_DATABASE_URL` first,
then `DATABASE_URL`. PostgreSQL integration tests are skipped when neither is
set.

## Devcontainer Setup

Install and start PostgreSQL inside the Debian devcontainer:

```bash
sudo apt-get update
sudo apt-get install -y postgresql postgresql-contrib
sudo service postgresql start
```

If `sudo` is not available and the container user is already root, omit `sudo`.

Create the local development role and database:

```bash
sudo -u postgres psql -c "CREATE ROLE arc3_agi LOGIN PASSWORD 'arc3_agi';"
sudo -u postgres createdb -O arc3_agi arc3_agi
```

If you want a separate database for tests:

```bash
sudo -u postgres createdb -O arc3_agi arc3_agi_test
export TEST_DATABASE_URL=postgresql://arc3_agi:arc3_agi@localhost:5432/arc3_agi_test
```

For normal experiment runs:

```bash
export DATABASE_URL=postgresql://arc3_agi:arc3_agi@localhost:5432/arc3_agi
```

Check connectivity:

```bash
psql "$DATABASE_URL" -c "select 1"
```

Run the focused experiment tests:

```bash
.venv/bin/python -m pytest arc3_agi/tests/test_experiment.py -q
```

## Claim Semantics

`maze_runner.run_experiment` uses `ExperimentStore.claim_experiment` before the
run starts.

- If the name is new, the row is inserted with `status = 'claimed'`, then moved
  to `running` while populations execute and `completed` after ingestion.
- If the name already exists with `status = 'completed'`, the runner skips and
  returns the existing experiment id.
- If the name exists with any other status, the runner raises an error instead
  of running a duplicate experiment.
- If the runner fails after claiming a name, the row is marked `failed` and the
  original exception is re-raised.

Historical DuckDB data in `experiments/runs.duckdb` is not migrated
automatically. Treat that as a separate one-off migration if those records need
to be preserved.
