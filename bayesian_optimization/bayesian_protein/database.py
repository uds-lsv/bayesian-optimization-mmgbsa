import dataclasses
import sqlite3
from pathlib import Path
from typing import List, Optional, Literal

from bayesian_protein.types import EmbeddingModel, Sampler, Simulator, Surrogate


@dataclasses.dataclass
class Experiment:
    n_iter: int
    embedding_model: EmbeddingModel
    data_path: str
    sampler: Sampler
    init_sampler: Literal["random", "closest"]
    surrogate: Surrogate
    protein: str
    seed: int
    batch_size: int
    init_batch_size: int
    output_file: Optional[str] = None
    n_clusters: Optional[int] = 1
    simulator: Optional[Simulator] = None
    bandit_confidence: Optional[float] = None
    job_id: Optional[int] = -1

    @staticmethod
    def sql_table() -> str:
        return """
        CREATE TABLE IF NOT EXISTS experiment (
            id INTEGER not null,
            n_iter INTEGER not null,
            batch_size INTEGER not null,
            init_batch_size INTEGER not null,
            embedding_model TEXT not null,
            data_path TEXT not null,
            sampler TEXT not null,
            init_sampler TEXT not null,
            surrogate TEXT not null,
            protein TEXT not null,
            seed INTEGER not null,
            output_file TEXT,
            n_clusters INTEGER,
            simulator TEXT,
            bandit_confidence REAL,
            job_id INTEGER,
            PRIMARY KEY("id" AUTOINCREMENT)
        );"""

    @staticmethod
    def sql_insert():
        return """
        INSERT INTO experiment (
            n_iter,
            batch_size,
            init_batch_size,
            embedding_model,
            data_path,
            sampler,
            init_sampler,
            surrogate,
            protein,
            seed,
            output_file,
            n_clusters,
            simulator,
            bandit_confidence,
            job_id
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        );
        """


@dataclasses.dataclass
class QueryResult:
    experiment_id: int
    molecule: str  # SMILES string of the molecule
    affinity: float  # Observed affinity
    prediction_mean: float  # Predicted affinity
    prediction_std: float  # Predicted affinity variance
    acquisition_score: float
    iteration: int  # At which iteration this molecule was picked
    cluster: int  # From which cluster this molecule was picked
    output_file_path: Optional[
        Path
    ]  # Where the output of the docking simulation is located
    is_validation_result: bool

    @staticmethod
    def sql_table() -> str:
        return """
        CREATE TABLE IF NOT EXISTS query_result (
            id INTEGER not null,
            experiment_id INTEGER not null,   -- Must map to an experiment.id
            molecule TEXT not null,
            affinity REAL not null,
            prediction_mean REAL not null,
            prediction_std REAL not null,
            acquisition_score REAL,  -- May be null for e.g. random acquisition or validation results
            iteration INTEGER not null,
            cluster INTEGER not null,
            output_file_path TEXT,
            is_validation_result INTEGER not null DEFAULT 0,

            FOREIGN KEY (experiment_id) REFERENCES experiment(id)
            PRIMARY KEY("id" AUTOINCREMENT)
        );
        """

    @staticmethod
    def sql_insert():
        return """
        INSERT INTO query_result (
            experiment_id,
            molecule,
            affinity,
            prediction_mean,
            prediction_std,
            acquisition_score,
            iteration,
            cluster,
            output_file_path,
            is_validation_result
        ) VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        );
        """


@dataclasses.dataclass
class ValidationSummaryResult:
    experiment_id: str
    samples: int
    iteration: int
    cluster: int
    mean_absolute_error: float
    mean_squared_error: float
    R2: float

    @staticmethod
    def sql_table() -> str:
        return """
            CREATE TABLE IF NOT EXISTS validation_summary (
                id INTEGER not null,
                experiment_id INTEGER not null,   -- Must map to an experiment.id
                samples INTEGER not null,
                iteration INTEGER not null,
                cluster INTEGER not null,
                mean_absolute_error REAL not null,
                mean_squared_error REAL not null,
                R2 REAL not null,

                FOREIGN KEY (experiment_id) REFERENCES experiment(id)
                PRIMARY KEY("id" AUTOINCREMENT)
            );
            """

    @staticmethod
    def sql_insert():
        return """
            INSERT INTO validation_summary (
                experiment_id,
                samples,
                iteration,
                cluster,
                mean_absolute_error,
                mean_squared_error,
                R2
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?
            );
            """


class Database:
    def __init__(self, dbpath: Path) -> None:
        self.dbpath = Path(dbpath).expanduser()
        self.dbpath.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(self.dbpath)

        with self.connection:
            cursor = self.connection.cursor()
            cursor.execute(Experiment.sql_table())
            cursor.execute(QueryResult.sql_table())
            cursor.execute(ValidationSummaryResult.sql_table())

    def insert(self, instance: dataclasses.dataclass) -> int:
        with self.connection:
            cursor = self.connection.cursor()
            cursor.execute(instance.sql_insert(), dataclasses.astuple(instance))
            return cursor.lastrowid

    def insert_many(self, instances: List[dataclasses.dataclass]):
        if not instances:
            return

        with self.connection:
            cursor = self.connection.cursor()
            values_list = [dataclasses.astuple(instance) for instance in instances]
            cursor.executemany(instances[0].sql_insert(), values_list)

    def insert_result(
        self,
        experiment_id: int,
        molecule: str,
        affinity: float,
        prediction_mean: float,
        prediction_std: float,
        acquisition_score: float,
        iteration: int,
        cluster: int,
        output_file_path: Optional[Path],
        is_validation_result: bool,
    ):
        with self.connection:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT EXISTS(SELECT 1 FROM experiment WHERE id = ?)", (experiment_id,)
            )
            exists = cursor.fetchone()[0]

        if not exists:
            raise ValueError(
                f"Invalid id. No experiment with {experiment_id=} exists in the database."
            )

        query_result = QueryResult(
            experiment_id=experiment_id,
            molecule=molecule,
            affinity=affinity,
            prediction_mean=prediction_mean,
            prediction_std=prediction_std,
            acquisition_score=acquisition_score,
            iteration=iteration,
            cluster=cluster,
            output_file_path=str(output_file_path)
            if output_file_path is not None
            else None,
            is_validation_result=is_validation_result,
        )

        return self.insert(query_result)

    def insert_experiment(
        self,
        n_iter: int,
        batch_size,
        init_batch_size,
        embedding_model: EmbeddingModel,
        data_path: Path,
        sampler: Sampler,
        init_sampler: Literal["closest", "random"],
        surrogate: Surrogate,
        protein: str,
        seed: int,
        output_file: Optional[Path] = None,
        n_clusters: Optional[int] = 1,
        simulator: Optional[Simulator] = None,
        bandit_confidence: Optional[float] = None,
        job_id: Optional[int] = -1,
    ):
        # Data Validation
        experiment = Experiment(
            n_iter,
            batch_size,
            init_batch_size,
            embedding_model,
            str(data_path),
            sampler,
            init_sampler,
            surrogate,
            protein,
            seed,
            str(output_file),
            n_clusters,
            simulator,
            bandit_confidence,
            job_id,
        )
        return self.insert(experiment)


if __name__ == "__main__":
    db = Database("~/.protein_ligand/database.sqlite")
