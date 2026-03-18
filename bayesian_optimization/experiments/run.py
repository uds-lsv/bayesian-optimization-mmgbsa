import functools
import os
import pathlib

import pandas as pd

from bayesian_protein.cli import parse_args
from bayesian_protein.embedding import embedding_factory
from bayesian_protein.optimizer import Optimizer
from bayesian_protein.types import (
    EmbeddingModel,
)


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

    results_dir = pathlib.Path(__file__).parent.parent / "runs"

    jobs = parse_args()
    for args in jobs:
        if args.surrogate == "constant" and args.sampler != "random":
            continue

        elif args.sampler == "random" and args.surrogate != "constant":
            continue
        
        db_path = str(results_dir / f"{args.out}.sqlite")
        optimizer = Optimizer(
            args,
            get_data_with_embeddings(
                embedding_model=args.embedding, csvfile=str(args.data)
            ).copy(),
            job_id=0,
            database_path=db_path,
        )
        print(db_path)
        optimizer.run()
