"""Experiment tracking store backed by DuckDB.

Provides :class:`ExperimentStore` for persisting per-generation fitness
statistics from parallel population runs and associating them with named,
described experiments.

Typical workflow::

    from arc3_agi.experiment import ExperimentStore

    store = ExperimentStore()                          # opens experiments/runs.duckdb
    eid = store.create_experiment(
        name="baseline",
        description="100 pops, default params",
        run_id="20260717T210555_6ae7c2",
        params={"max_generations": 1000, "population_size": 100},
    )
    store.ingest_run(eid, Path("runs/20260717T210555_6ae7c2"))
    df = store.load_stats(eid)   # pandas DataFrame
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

DEFAULT_DB_PATH: Path = Path("experiments") / "runs.duckdb"

_SCHEMA_SQL = """
CREATE SEQUENCE IF NOT EXISTS experiments_id_seq START 1;

CREATE TABLE IF NOT EXISTS experiments (
    id          BIGINT PRIMARY KEY DEFAULT nextval('experiments_id_seq'),
    name        TEXT    NOT NULL,
    description TEXT,
    run_id      TEXT,
    created_at  TIMESTAMP NOT NULL,
    params_json TEXT
);

CREATE TABLE IF NOT EXISTS generation_stats (
    experiment_id BIGINT  NOT NULL,
    pop_id        INTEGER NOT NULL,
    generation    INTEGER NOT NULL,
    min_fitness   REAL    NOT NULL,
    max_fitness   REAL    NOT NULL,
    mean_fitness  REAL    NOT NULL,
    duration_s    REAL    NOT NULL,
    PRIMARY KEY (experiment_id, pop_id, generation)
);
"""


class ExperimentStore:
    """DuckDB-backed store for experiment metadata and per-generation stats.

    Parameters
    ----------
    db_path:
        Path to the DuckDB database file.  Created (along with parent
        directories) if it does not already exist.
        Defaults to :data:`DEFAULT_DB_PATH` (``experiments/runs.duckdb``).
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        _conn: duckdb.DuckDBPyConnection | None = duckdb.connect(str(self._path))
        try:
            _conn.execute(_SCHEMA_SQL)
        except Exception:
            _conn.close()
            raise
        assert _conn is not None
        self._conn: duckdb.DuckDBPyConnection = _conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        description: str = "",
        run_id: str = "",
        params: dict[str, Any] | None = None,
    ) -> int:
        """Insert a new experiment record and return its id.

        Parameters
        ----------
        name:
            Short human-readable name (e.g. ``"baseline"``).
        description:
            Free-text description of the experiment's purpose and settings.
        run_id:
            The run identifier (matches the ``runs/<run_id>/`` directory).
        params:
            Dictionary of configuration parameters to store as JSON.

        Returns
        -------
        int
            The auto-assigned experiment id.
        """
        params_json = json.dumps(params or {})
        created_at = datetime.now(tz=timezone.utc)
        row = self._conn.execute(
            """
            INSERT INTO experiments (name, description, run_id, created_at, params_json)
            VALUES (?, ?, ?, ?, ?)
            RETURNING id
            """,
            [name, description, run_id, created_at, params_json],
        ).fetchone()
        return int(row[0])  # type: ignore[index]

    def get_experiment_id_by_name(self, name: str) -> int | None:
        """Return the id for an existing experiment named *name*, if any."""
        row = self._conn.execute(
            """
            SELECT id
            FROM experiments
            WHERE name = ?
            ORDER BY id
            LIMIT 1
            """,
            [name],
        ).fetchone()
        return int(row[0]) if row is not None else None

    def ingest_run(self, experiment_id: int, run_dir: Path | str) -> int:
        """Read all ``fitness_history.json`` files under *run_dir* and insert
        their generation stats into the database.

        Each population subprocess writes a ``fitness_history.json`` directly
        inside its ``pop_N/`` directory (not inside the timestamped
        subdirectory).  This method globs for those files under *run_dir*.

        Parameters
        ----------
        experiment_id:
            Id returned by :meth:`create_experiment`.
        run_dir:
            Path to the run directory, e.g. ``runs/20260717T210555_6ae7c2``.

        Returns
        -------
        int
            Total number of generation-stat rows inserted.
        """
        run_dir = Path(run_dir)
        rows: list[tuple[int, int, int, float, float, float, float]] = []

        for history_file in sorted(run_dir.glob("pop_*/fitness_history.json")):
            with history_file.open() as fh:
                data = json.load(fh)
            pop_id: int = data["pop_id"]
            for entry in data["history"]:
                rows.append(
                    (
                        experiment_id,
                        pop_id,
                        entry["generation"],
                        entry["min_fitness"],
                        entry["max_fitness"],
                        entry["mean_fitness"],
                        entry["duration_s"],
                    )
                )

        if rows:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO generation_stats
                    (experiment_id, pop_id, generation,
                     min_fitness, max_fitness, mean_fitness, duration_s)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def list_experiments(self) -> pd.DataFrame:
        """Return a DataFrame of all experiments with summary counts.

        Columns: ``id``, ``name``, ``description``, ``run_id``, ``created_at``,
        ``pop_count``, ``gen_count``, ``params_json``.
        """
        return self._conn.execute("""
            SELECT
                e.id,
                e.name,
                e.description,
                e.run_id,
                e.created_at,
                COUNT(DISTINCT gs.pop_id)    AS pop_count,
                MAX(gs.generation)           AS gen_count,
                e.params_json
            FROM experiments e
            LEFT JOIN generation_stats gs ON gs.experiment_id = e.id
            GROUP BY e.id, e.name, e.description, e.run_id, e.created_at, e.params_json
            ORDER BY e.id
            """).df()

    def load_stats(self, experiment_id: int) -> pd.DataFrame:
        """Return all generation-level stats for *experiment_id*.

        Columns: ``pop_id``, ``generation``, ``min_fitness``, ``max_fitness``,
        ``mean_fitness``, ``duration_s``.
        """
        return self._conn.execute(
            """
            SELECT pop_id, generation, min_fitness, max_fitness, mean_fitness, duration_s
            FROM generation_stats
            WHERE experiment_id = ?
            ORDER BY pop_id, generation
            """,
            [experiment_id],
        ).df()

    def delete_experiment(self, experiment_id: int) -> None:
        """Remove an experiment and all its generation stats.

        Parameters
        ----------
        experiment_id:
            Id of the experiment to delete.
        """
        self._conn.execute(
            "DELETE FROM generation_stats WHERE experiment_id = ?", [experiment_id]
        )
        self._conn.execute("DELETE FROM experiments WHERE id = ?", [experiment_id])

    def close(self) -> None:
        """Close the underlying DuckDB connection."""
        if self._conn is not None:
            self._conn.close()

    def __enter__(self) -> "ExperimentStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
