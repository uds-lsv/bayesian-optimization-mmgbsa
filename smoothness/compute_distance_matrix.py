import argparse
import pathlib


import numpy as np
from scipy.spatial.distance import pdist, squareform


parser = argparse.ArgumentParser(description="Given an array of embeddings in .npy format computes the pairwise euclidean distance and stores the output in vector-form.")
parser.add_argument("embedding_file", type=pathlib.Path)
parser.add_argument("outdir", type=pathlib.Path)

args = parser.parse_args()

assert args.embedding_file.exists()

# May take a lot of RAM
embeddings = np.load(args.embedding_file)
dist = pdist(embeddings, metric="euclidean")

np.save(args.outdir / f"{args.embedding_file.stem}_distance.npy", dist)
