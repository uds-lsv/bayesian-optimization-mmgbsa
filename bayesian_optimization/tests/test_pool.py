import unittest.mock
from unittest import TestCase

import numpy as np
import pandas as pd

from bayesian_protein import ClusteredLigandPools


class TestClusteredLigandPools(TestCase):
    def setUp(self):
        # Cluster 0 is E F D
        # Cluster 1 is B C A
        self.df = pd.DataFrame(
            {
                "smiles": ["A", "B", "C", "D", "E", "F"],
                "embedding": [
                    np.array([0.3, 0.3]),
                    np.array([0.1, 0.1]),
                    np.array([0, 0]),
                    np.array([1.3, 1.3]),
                    np.array([1.1, 1.1]),
                    np.array([1, 1]),
                ],
            }
        )

        self.pool = ClusteredLigandPools(self.df, k=2, seed=0)
        self.model = unittest.mock.Mock()

    def test_not_all_columns_raises(self):
        columns = ["smiles", "embedding"]
        for col in columns:
            with self.subTest(column=col):
                with self.assertRaises(ValueError) as context:
                    ClusteredLigandPools(pd.DataFrame({"smiles": []}), 1)
                self.assertEqual(
                    "The dataframe needs to contain columns 'smiles' and 'embedding'.",
                    str(context.exception),
                )

    def test_validate_cluster_id(self):
        for cluster_id in [-1, 5]:
            with self.assertRaises(ValueError) as context:
                self.pool.get_cluster(cluster_id)
            self.assertEqual(
                f"Value of cluster_id ({cluster_id}) is not in the valid range [0, 2).",
                str(context.exception),
            )

    def test_cluster_is_sorted(self):
        # We check that after clustering the clusters are sorted.
        for k in [1, 2]:
            with self.subTest(k=k):
                pool = ClusteredLigandPools(self.df, seed=0, k=k)
                for c in range(k):
                    cluster = pool.get_cluster(c)
                    self.assertTrue(
                        cluster["distance_to_centroid"].is_monotonic_increasing
                    )

    def test_get_cluster(self):
        self.pool.get_cluster(0)["smiles"].equals(self.df["smiles"].iloc[3:6])
        self.pool.get_cluster(1)["smiles"].equals(self.df["smiles"].iloc[0:3])

    def test_sample_raises_with_incorrect_sampler(self):
        with self.assertRaises(ValueError) as context:
            self.pool.sample("incorrect-sampler", 0, self.model)
        # Rest of message is irrelevant
        self.assertTrue(
            str(context.exception).startswith("Invalid sampler incorrect-sampler.")
        )

    def test_sample(self):
        cases = [("random", ["D", "C"]), ("closest", ["E", "B"])]  # rng is fix
        for sampler, expected_smiles in cases:
            with self.subTest(sampler=sampler):
                for cluster_id, expected in zip(range(self.pool.k), expected_smiles):
                    _idx, sample = self.pool.sample(sampler, cluster_id, self.model)
                    self.assertEqual(expected, sample["smiles"])
                    self.assertTrue(sample["queried"])
                    self.assertTrue(self.pool._data.loc[sample.name]["queried"])

    def test_sample_expected_improvement_queried_empty_raises(self):
        with self.assertRaises(ValueError) as context:
            self.pool.sample("expected-improvement", 0, self.model)
        self.assertEqual(
            "No samples have been queried. Expected improvement can only be used "
            "once a molecule from this cluster has been labeled.",
            str(context.exception),
        )

    def test_sample_expected_improvement(self):
        expected_smiles = ["F", "C"]
        self.model.expected_improvement_batch = unittest.mock.Mock(
            return_value=np.array([3, -1])
        )
        for cluster_id, expected in zip(range(self.pool.k), expected_smiles):
            # Before we can run EI we need one reference sample
            _idx, init = self.pool.sample("closest", cluster_id, self.model)
            self.pool.set_value(init.name, 3)

            _idx, sample = self.pool.sample(
                "expected-improvement", cluster_id, self.model
            )
            self.assertEqual(expected, sample["smiles"])
            self.assertTrue(sample["queried"])
            self.assertTrue(self.pool._data.loc[sample.name]["queried"])

    def test_sample_empty_cluster(self):
        for _ in range(3):
            self.pool.sample("closest", 0, self.model)

        with self.assertRaises(ValueError) as context:
            self.pool.sample("closest", 0, self.model)
        self.assertEqual(
            "Cluster 0 contains no unlabeled samples.", str(context.exception)
        )

    def test_data_contains_duplicates_raises(self):
        with self.assertRaises(ValueError) as context:
            df = pd.concat(
                [
                    self.df,
                    pd.DataFrame(
                        {"smiles": ["A"], "embeddings": [np.array([0.1, 0.1])]}
                    ),
                ]
            )
            ClusteredLigandPools(df, 1)
        self.assertEqual(
            str(context.exception),
            "The dataframe contains duplicated entries in the 'smiles' column.",
        )
