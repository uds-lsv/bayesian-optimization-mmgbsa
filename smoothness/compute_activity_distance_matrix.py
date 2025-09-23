import argparse
import pathlib

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform


parser = argparse.ArgumentParser(description="Computes the absolute pairwise distance between all activity values and stores the result in vector-form")
parser.add_argument("activity", choices=["vina", "mmgbsa"])
parser.add_argument("outdir", type=pathlib.Path)

args = parser.parse_args()


df = pd.read_csv(f"../data/MCL1-{args.activity}.csv")
activity = df["target"].values

# Pairwise absolute distance
dist = pdist(activity.reshape(-1, 1), metric='cityblock')
np.save(args.outdir / f"{args.activity}_distance.npy", dist)