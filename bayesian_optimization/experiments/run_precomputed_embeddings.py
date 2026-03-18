import functools
import os
import pathlib
import pickle

import pandas as pd

from bayesian_protein.cli import parse_args
from bayesian_protein.embedding import embedding_factory
from bayesian_protein.optimizer import Optimizer
from bayesian_protein.types import (
    EmbeddingModel,
)



if __name__ == "__main__":
    # os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Huggingface tokenizers

    results_dir = pathlib.Path(__file__).parent.parent / "runs"

    with open("embeddings_finetuned.pkl", "rb") as f:
        e = pickle.load(f)
        finetuned = pd.DataFrame(e).set_index("name")

    with open("embeddings_untuned.pkl", "rb") as f:
        not_finetuned = pd.DataFrame(pickle.load(f)).set_index("name")



    jobs = parse_args()
    for embedding, name in [(finetuned, "finetuned"), (not_finetuned, "untuned")]:
        for args in jobs:
            if args.surrogate == "constant" and args.sampler != "random":
                continue

            elif args.sampler == "random" and args.surrogate != "constant":
                continue

            data = pd.read_csv(str(args.data)).set_index("name")
            # We assume that embeddings are already ordered in the same way
            data = data.join(embedding).drop(columns="pb_score").reset_index(drop=True).rename(columns={"embeddings": "embedding"})

            args.embedding = f"precomputed-{name}"
            db_path = str(results_dir / f"{args.out}.sqlite")
            optimizer = Optimizer(
                args,
                data,
                job_id=0,
                database_path=db_path,
            )
            print(db_path)
            optimizer.run()