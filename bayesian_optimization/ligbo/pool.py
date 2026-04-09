import warnings
from functools import wraps
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ligbo.surrogates import BaseSurrogate
from ligbo.types import VALID_SAMPLERS, Sampler


def validate_cluster_id(method):
    """
    Decorator to check if the cluster id is in the range [0, self.k)
    :param method: The method to decorate
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        cluster_id = kwargs.get("cluster_id", args[0] if args else None)
        if cluster_id is None or not 0 <= cluster_id < self.k:
            raise ValueError(
                f"Value of cluster_id ({cluster_id}) is not in the valid range [0, {self.k})."
            )
        return method(self, *args, **kwargs)

    return wrapper


class ClusteredLigandPools:
    def __init__(self, data: pd.DataFrame, seed: int, k: int = 1):
        """
        Pool of possible molecules split into k clusters.
        :param data: DataFrame with at least smiles and embedding column. The embedding column should contain numpy
            arrays of dimension (D,). The 'target' column is used to save measured affinities
        :param k: Number of clusters
        """
        if not all(c in data.columns for c in ["smiles", "embedding"]):
            raise ValueError(
                "The dataframe needs to contain columns 'smiles' and 'embedding'."
            )

        if data["smiles"].duplicated().any():
            warnings.warn(
                "The dataframe contains duplicated entries in the 'smiles' column."
            )

        self._data = data
        self._data["queried"] = False

        self.k = k
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.cluster()
        self.embeddings = np.row_stack(self._data["embedding"].values)

    def cluster(self):
        """
        Clusters the pool using k-Means and Euclidean distances.
        The cluster assignments are stored in the dataframe and each cluster is
        sorted by ascending distance to the centroid.
        """
        embeddings = np.row_stack(self._data["embedding"].values)
        kmeans = KMeans(n_clusters=self.k, random_state=self.seed, n_init="auto")
        kmeans.fit(embeddings)
        self._data["cluster"] = kmeans.labels_

        # Calculate the distance of each point to its cluster centroid
        centroids = kmeans.cluster_centers_
        distances = np.linalg.norm(embeddings - centroids[kmeans.labels_], axis=1)
        self._data["distance_to_centroid"] = distances

        # Split the data into k pools based on clusters and sort by distance within each cluster
        for i in range(self.k):
            self.sort_cluster(i)

    @property
    def cluster_ids(self) -> List[int]:
        """
        Possible cluster ids
        """
        return [int(c) for c in self._data["cluster"].unique()]

    @validate_cluster_id
    def sort_cluster(self, cluster_id: int):
        """
        Sorts the cluster with the given id in place.
        :param cluster_id: Number of the cluster must be in [0, k)
        """
        cluster = self._data[self._data["cluster"] == cluster_id]
        cluster = cluster.sort_values(by="distance_to_centroid")
        self._data = pd.concat(
            [self._data.drop(cluster.index), cluster], ignore_index=True
        )

    @validate_cluster_id
    def get_cluster(self, cluster_id: int) -> pd.DataFrame:
        """
        :param cluster_id: cluster_id: Number of the cluster must be in [0, k)
        :return: Returns the subset of the data belonging to this cluster
        """
        pool = self._data[self._data["cluster"] == cluster_id]
        return pool

    def sample(self, by: Sampler, cluster_id: int, model: BaseSurrogate, size: int, kappa: float = 2.0):
        unlabeled = self._data[
            (self._data["cluster"] == cluster_id) & (~self._data["queried"])
        ]

        if len(unlabeled) == 0:
            raise ValueError(f"Cluster {cluster_id} contains no unlabeled samples.")

        size = min(size, len(unlabeled))

        if by == "random":
            idx = self.rng.choice(unlabeled.index, size=size, replace=False)
            score = [None for _ in idx]
        elif by == "closest":
            assert unlabeled[
                "distance_to_centroid"
            ].is_monotonic_increasing, "Pool is not sorted"
            idx = unlabeled.index[:size]
            score =  unlabeled["distance_to_centroid"][:size]

        elif by == "xth-closest":
            warnings.warn(f"xth-closest will acquire only the {size}th closest input to the medoid.")
            assert unlabeled[
                "distance_to_centroid"
            ].is_monotonic_increasing, "Pool is not sorted"
            # Make sure we return DataFrames, using just size returns a Series
            # and then errors when iterating over results["smiles"] because that
            # then returns just the string
            idx = unlabeled.index[size:size+1]
            score = unlabeled["distance_to_centroid"][size:size+1]

        elif by == "stochastic-closest":
            assert unlabeled[
                "distance_to_centroid"
            ].is_monotonic_increasing, "Pool is not sorted"
            # size is reused as neighborhood K (top-K nearest); return 1 random molecule from it
            k = min(size, len(unlabeled))
            neighborhood_idx = unlabeled.index[:k]
            chosen = self.rng.choice(neighborhood_idx, size=1, replace=False)
            idx = pd.Index(chosen)
            score = unlabeled.loc[chosen, "distance_to_centroid"]

        elif by == "greedy":
            embeddings = self.embeddings[unlabeled.index.values]
            prediction_mean, _ = model.forward(embeddings, unlabeled["smiles"].tolist())
            # We want the k items that have the smallest predicted value (=best affinity)
            arr_idx = np.argpartition(prediction_mean, size)[:size]
            idx = unlabeled.iloc[arr_idx].index
            score = prediction_mean[arr_idx]
        elif by == "expected-improvement":
            cluster = self.get_cluster(cluster_id)
            labeled = cluster[cluster["queried"]]
            if len(labeled) == 0:
                raise ValueError(
                    "No samples have been queried. Expected improvement can only be used once a molecule from this cluster has been labeled."
                )
            best_seen = labeled["target"].min()
            embeddings = self.embeddings[unlabeled.index.values]
            scores = model.expected_improvement_batch(
                embeddings, best_seen, unlabeled["smiles"].tolist()
            )
            scores[scores < 0] = 0
            arr_idx = np.argpartition(scores, -size)[-size:]
            idx = unlabeled.iloc[arr_idx].index
            score = scores[arr_idx]
        elif by == "ucb":
            embeddings = self.embeddings[unlabeled.index.values]
            scores = model.ucb_batch(embeddings, kappa, unlabeled["smiles"].tolist())
            arr_idx = np.argpartition(scores, size)[:size]  # smallest LCB = most promising
            idx = unlabeled.iloc[arr_idx].index
            score = scores[arr_idx]
        elif by == "diverse":
            # Greedy MaxMin (furthest-first): iteratively add the molecule that is
            # furthest from all already-selected molecules, maximising coverage of
            # embedding space. Seed with index 0 (closest to centroid, since clusters
            # are sorted by ascending distance_to_centroid).
            #
            # Invariant: dist_to_nearest_selected[i] = distance from candidate i to
            # its nearest already-selected molecule. The candidate with the largest
            # value is the least-covered one and is picked next. Already-selected
            # candidates are set to -inf so they are never re-picked.
            embeddings = self.embeddings[unlabeled.index.values]  # (n, d)
            selected = [0]  # seed: closest to centroid
            dist_to_nearest_selected = np.linalg.norm(embeddings - embeddings[0], axis=1)  # (n,)
            dist_to_nearest_selected[0] = -np.inf  # exclude from future selection
            while len(selected) < size:
                # Molecule furthest from all selected so far
                next_idx = int(np.argmax(dist_to_nearest_selected))
                selected.append(next_idx)
                dist_to_nearest_selected[next_idx] = -np.inf
                # Update: each candidate's nearest selected may now be next_idx
                dist_to_new = np.linalg.norm(embeddings - embeddings[next_idx], axis=1)  # (n,)
                dist_to_nearest_selected = np.minimum(dist_to_nearest_selected, dist_to_new)  # (n,)
            idx = unlabeled.iloc[selected].index
            score = [None] * size

        elif by == "explore":
            # Pick the k items with the largest variance.
            # If the output follows a gaussian distribution (e.g. regression) the variance
            # is equal to the expected mean squared error of a point x, hence choosing the
            # points with the largest variance corresponds to choosing the points with the
            # largest expected error. At the same time the variance is equal to 1/(2pi e)exp(2H(y|x))
            # (see e.g. T.Cover - Elements of Information Theory p.255)
            # Hence corresponds to choosing the points with the largest entropy.
            embeddings = self.embeddings[unlabeled.index.values]
            _, prediction_std = model.forward(embeddings, unlabeled["smiles"].tolist())
            arr_idx = np.argpartition(prediction_std, -size)[-size:]
            idx = unlabeled.iloc[arr_idx].index
            score = prediction_std[arr_idx]
        else:
            raise ValueError(
                f"Invalid sampler {by}. Valid choices are: {', '.join(VALID_SAMPLERS)}"
            )

        self.set_queried(idx)
        return idx, self._data.loc[idx], score

    def set_queried(self, index: int):
        self._update(index, "queried", True)

    def set_value(self, index: int, value: float):
        self._update(index, "target", value)

    def _update(self, index: int, key: str, value):
        self._data.loc[index, key] = value

    def get(self, idx: int):
        return self._data.loc[idx]

    @validate_cluster_id
    def cluster_is_empty(self, cluster_id):
        return (
            len(
                self._data[
                    (self._data["cluster"] == cluster_id) & (~self._data["queried"])
                ]
            )
            == 0
        )
