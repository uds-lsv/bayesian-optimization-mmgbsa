import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from bayesian_protein.bandit import MultiarmedBanditPlayer
from bayesian_protein.cli import CommandLineArgs
from bayesian_protein.database import Database, QueryResult, ValidationSummaryResult
from bayesian_protein.embedding import embedding_factory
from bayesian_protein.pool import ClusteredLigandPools
from bayesian_protein.simulation import SimulatorBase
from bayesian_protein.surrogates import surrogate_factory
from bayesian_protein.types import Sampler


class Optimizer:
    def setup_logger(self):
        logger = logging.getLogger(f"bayesian_protein ({self.job_id})")
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "[%(levelname)s] %(asctime)s - %(name)s: %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        else:
            logger = logging.getLogger(f"bayesian_protein ({self.job_id})")
        return logger

    def __init__(
        self,
        args: CommandLineArgs,
        data: pd.DataFrame,
        database_path: Optional[str] = os.path.join(
            os.path.dirname(__file__), "../protein_ligand.sqlite"
        ),
        job_id: Optional[int] = -1,
    ):
        """

        :param args: Configuration see CommandLineArgs
        :param data: DataFrame with at least smiles and embedding column. The embedding column should contain numpy
            arrays of dimension (D,). The 'target' column is used to save measured affinities
        :param database_path: Path to the output database
        :param job_id: Optional integer specifying the current job ID on the cluster. Helps to distinguish between
            different jobs in the logs.
        """
        self.args = args
        self.data = data
        self.pool = ClusteredLigandPools(self.data, args.seed, k=args.cluster)
        self.models = {
            cluster_id: surrogate_factory(args.surrogate)
            for cluster_id in self.pool.cluster_ids
        }
        self.bandit = MultiarmedBanditPlayer(self.pool.cluster_ids, args.confidence)
        self.db = Database(dbpath=database_path)
        self.job_id = job_id

        self.simulator = args.get_simulator(self.data)

        self.logger = self.setup_logger()

    def run(self):
        self.store_experiment()
        with self.simulator as simulator:
            self.logger.info(f"Initializing {self.args.cluster:>3} clusters.")
            self.initialize_cluster(simulator)

            for i in range(self.args.n_iter):
                t0 = time.time()

                cluster_id = self.bandit.choose()
                self.logger.debug(f"({i:0>4}) Chose cluster {cluster_id} based on UCB.")
                self.query(
                    cluster_id, simulator, iteration=i, sampler=self.args.sampler, size=self.args.sample_size
                )

                if self.pool.cluster_is_empty(cluster_id):
                    self.bandit.remove(cluster_id)

                self.validate(iteration=i, cluster_id=cluster_id)

                self.logger.info(f"{i:0>4} Iteration took {time.time() - t0:.2f}s")

    def store_experiment(self):
        self.experiment_id = self.db.insert_experiment(
            n_iter=self.args.n_iter,
            batch_size=self.args.sample_size,
            init_batch_size=self.args.init_sample_size,
            embedding_model=self.args.embedding,
            output_file=self.args.out,
            data_path=self.args.data,
            sampler=self.args.sampler,
            init_sampler=self.args.init_sampler,
            surrogate=self.args.surrogate,
            protein=self.args.protein,
            seed=self.args.seed,
            n_clusters=self.args.cluster,
            simulator=self.args.simulate,
            bandit_confidence=self.args.confidence,
            job_id=self.job_id,
        )
        self.logger.info(f"Experiment has id {self.experiment_id}.")

    def initialize_cluster(self, simulator: SimulatorBase):
        """
        Simulates the molecule that is closest to the centroid in order to initialize the bandit
        :param simulator:
        :return:
        """
        for cluster_id in self.pool.cluster_ids:
            self.query(
                cluster_id,
                simulator,
                iteration=-1,
                sampler=self.args.init_sampler,
                size=self.args.init_sample_size
            )

    def query(
        self,
        cluster_id: int,
        simulator: SimulatorBase,
        iteration: int,
        sampler: Sampler,
        size: int
    ):
        model = self.models[cluster_id]

        t0 = time.time()
        idx, result, acq_scores = self.pool.sample(
            sampler, cluster_id, model, size
        )
        self.logger.debug(
            f"({iteration:0>4}) Chosen sample {result['smiles']}: {time.time() - t0:.2f}s"
        )
        affinities, paths = simulator.simulate_many(result["smiles"])
        self.logger.debug(
            f"({iteration:0>4}) Simulated {result['smiles']}: {affinities}"
        )

        self.pool.set_value(idx, affinities)
        embedding = np.stack(result["embedding"].values)
        # TODO: If using expected improvement this calculates the forward pass twice
        if iteration > -1:  # Initial iteration
            prediction_mean, prediction_std = self.models[cluster_id].forward(
                embedding, result["smiles"].tolist()
            )
        else:
            prediction_mean = [np.inf] * len(result["smiles"])
            prediction_std = [0] * len(result["smiles"])

        self.logger.debug(
            f"({iteration:0>4}) Predicted {result['smiles']}: {prediction_mean}"
        )

        if self.args.sample_size == 1:
            embedding = embedding.reshape(1, -1)

        model.update(embedding, affinities, result["smiles"].tolist())
        self.bandit.update(cluster_id, sum(affinities))
        self.db.insert_many(
            [
                QueryResult(
                    experiment_id=self.experiment_id,
                    molecule=molecule,
                    affinity=affinity,
                    prediction_mean=float(mean),
                    prediction_std=float(std),
                    acquisition_score=float(score) if score is not None else None,  # None when random sampling
                    iteration=iteration,
                    cluster=cluster_id,
                    output_file_path=str(path) if path is not None else None,
                    is_validation_result=False,
                )
                for (molecule, affinity, path, mean, std, score) in zip(
                    result["smiles"], affinities, paths, prediction_mean, prediction_std, acq_scores
                )
            ]
        )

        return embedding, affinities

    def validate(self, iteration, cluster_id):
        model = self.models[cluster_id]
        cluster = self.pool.get_cluster(cluster_id)
        unlabeled = cluster[~cluster["queried"]]
        self.logger.debug(f"({iteration:0>4}) Validating on {len(unlabeled)} molecules")

        # Predict
        embeddings = self.pool.embeddings[unlabeled.index.values]
        val_mean, val_std = model.forward(embeddings, unlabeled["smiles"].tolist())

        # R2 score is not well defined for less than 2 samples
        if len(unlabeled) < 2:
            r2 = -np.inf
        else:
            r2 = r2_score(unlabeled["target"], val_mean)
        # Always store a summary and only store all results
        # if explicitly asked
        summary = ValidationSummaryResult(
            experiment_id=self.experiment_id,
            samples=len(unlabeled),
            iteration=iteration,
            cluster=cluster_id,
            mean_absolute_error=mean_absolute_error(unlabeled["target"], val_mean),
            mean_squared_error=mean_squared_error(unlabeled["target"], val_mean),
            R2=r2,
        )
        self.logger.debug(f"({iteration:0>4}) {summary}")
        self.db.insert(summary)

        if self.args.validate:
            results = []
            for idx in range(len(unlabeled)):
                sample = unlabeled.iloc[idx]
                results.append(
                    QueryResult(
                        self.experiment_id,
                        sample["smiles"],
                        sample["target"],
                        float(val_mean[idx].item()),
                        float(val_std[idx].item()),
                        acquisition_score=None,
                        iteration=iteration,
                        cluster=cluster_id,
                        output_file_path=None,
                        is_validation_result=True,
                    )
                )

            self.db.insert_many(results)


def create_optimizer_with_embeddings(job_args: CommandLineArgs, *args, **kwargs):
    """
    Helper function to create the optimizer from the command line arguments
    Useful if the embeddings are only needed for this run
    """
    data = pd.read_csv(job_args.data)
    embedder = embedding_factory(job_args.embedding, batch_size=256)
    embeddings = embedder(data["smiles"].tolist())
    data.loc[:, "embedding"] = embeddings.tolist()

    return Optimizer(job_args, embeddings, *args, **kwargs)
