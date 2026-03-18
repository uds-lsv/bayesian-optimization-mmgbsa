import argparse
import functools
import itertools
import logging
import operator
import os
import time

import pandas as pd

from bayesian_protein.cli import CommandLineArgs
from bayesian_protein.embedding import embedding_factory
from bayesian_protein.optimizer import Optimizer
from bayesian_protein.types import (
    VALID_EMBEDDING_MODELS,
    VALID_SAMPLERS,
    VALID_SURROGATE_MODELS,
    EmbeddingModel,
)
from config import DATA_DIR, RESULTS_DIR
from utils.chembl import load_chembl, preprocess, top_k

CHEMBL_PATH = DATA_DIR / "processed" / "activities-chembl33.csv"

parser = argparse.ArgumentParser()
parser.add_argument(
    "job_id", type=int, help="Cluster job id, helpful for finding logs.", default=0
)
parser.add_argument(
    "--top-k",
    required=True,
    type=int,
    help="For how many proteins bayesian optimization should be run. Selects the k proteins with the most measurements in ChEMBL.",
)
parser.add_argument(
    "--chembl-path",
    type=str,
    default=CHEMBL_PATH,
    help=f"Path to the chembl activities .csv file. Defaults to {CHEMBL_PATH}.",
)
parser.add_argument(
    "--validate",
    default=False,
    action="store_true",
    help="Calculate loss on not datapoints that have not been queried. Only availalbe when used with premeasured datapoints.",
)
parser.add_argument("--seed", type=int, help="Seed for reproducibility.", default=42)


@functools.lru_cache(maxsize=None)
def get_data_with_embeddings(embedding_model: EmbeddingModel, csvfile: str):
    """
    Calculate embeddings for molecules in the csv file. Molecules are expected to be in SMILES format and
    the file should contain a column named "smiles"
    """
    data = pd.read_csv(csvfile)
    embedder = embedding_factory(embedding_model, batch_size=256)
    embeddings = embedder(data["smiles"].tolist())
    data.loc[:, "embedding"] = embeddings.tolist()
    return data


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Huggingface tokenizers
    args = parser.parse_args()

    if not os.path.exists(args.chembl_path):
        raise ValueError(f"{args.chembl_path} does not exist.")

    chembl = load_chembl(CHEMBL_PATH)
    chembl = preprocess(chembl)

    subsets = top_k(chembl, k=args.top_k)

    data_paths = []
    for subset in subsets:
        subset_data = subset.copy()
        assert "smiles" in subset.columns
        subset_data.loc[:, "target"] = -subset_data["value"]

        # Protein
        protein_name = subset_data["protein_chembl_id"].unique()[0]
        subset_path = DATA_DIR / "processed" / (protein_name + ".csv")
        data_paths.append(subset_path)
        if not subset_path.exists():
            subset_data[["smiles", "target"]].to_csv(subset_path, index=False)
        else:
            logging.info(f"File {subset_path} already exists")

    COMBINATIONS = {
        "n_iter": [100],
        "confidences": [0.1, 0.5, 1, 2, 5],
        "n_clusters": [1, 2, 3, 4, 5, 10, 20],
        "surrogates": VALID_SURROGATE_MODELS,
        "embedding_models": VALID_EMBEDDING_MODELS,
        "samplers": VALID_SAMPLERS,
    }

    jobs_to_run = []

    for parameters in [
        dict(zip(COMBINATIONS.keys(), vals))
        for vals in itertools.product(*COMBINATIONS.values())
    ]:
        if parameters["surrogates"] == "constant" and parameters["samplers"] not in (
            "closest",
            "random",
        ):
            continue
        elif (
            parameters["surrogates"] != "constant"
            and parameters["samplers"] != "expected-improvement"
        ):
            continue

        # One job with these parameters for all the proteins
        for path in data_paths:
            job_args = CommandLineArgs(
                n_iter=100,
                embedding=parameters["embedding_models"],
                out=None,
                data=path,
                sampler=parameters["samplers"],
                surrogate=parameters["surrogates"],
                protein=path.stem,
                cluster=parameters["n_clusters"],
                simulate=None,
                confidence=parameters["confidences"],
                validate=args.validate,
                seed=args.seed,
            )
            jobs_to_run.append(job_args)

    # If we run the jobs in the order embedding-model -> data, we can reuse the embeddings
    jobs_to_run.sort(key=operator.attrgetter("embedding", "data"))
    for i, job_args in enumerate(jobs_to_run):
        print(f"{i:>4}/{len(jobs_to_run)}", job_args)
        start = time.time()

        optimizer = Optimizer(
            job_args,
            get_data_with_embeddings(
                embedding_model=job_args.embedding, csvfile=str(job_args.data)
            ).copy(),
            job_id=args.job_id,
            # SQLite does not like concurrent accesses...
            # If we want to run two datasets at the same time we could run into problems.
            # Therefore, we create on database for each job.
            # If required one can merge the databases
            database_path=RESULTS_DIR / f"protein_ligand_{args.job_id}_chembl.sqlite",
        )
        optimizer.run()
        print(f"{i:0>4}: Finished after {time.time() - start:.3f}s.")
