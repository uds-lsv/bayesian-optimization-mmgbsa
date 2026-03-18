import pathlib
import tempfile
from unittest import TestCase

import numpy as np
import pandas as pd

from bayesian_protein.simulation import DatasetSimulator, SminaSimulator
from bayesian_protein import MANDATORY_COLUMNS


class TestDatasetSimulator(TestCase):
    def setUp(self):
        self.incomplete_data = pd.DataFrame({"smiles": ["A"], "target": [np.nan]})

    def test__check_data_missing_column(self):
        for col in MANDATORY_COLUMNS:
            with self.subTest(column=col):
                with self.assertRaises(ValueError) as context:
                    data = self.incomplete_data.drop(columns=col)
                    DatasetSimulator(data)

                self.assertEqual(
                    str(context.exception),
                    f"Dataset is missing mandatory columns: {col}.",
                )

    def test_data_has_nan_values_raises(self):
        with self.assertRaises(ValueError) as context:
            DatasetSimulator(self.incomplete_data)
        self.assertEqual(str(context.exception), "The dataset contains NaN entries.")

    def test_data_non_negative_values_raises(self):
        with self.assertRaises(ValueError) as context:
            self.incomplete_data.loc[0, "target"] = 1
            DatasetSimulator(self.incomplete_data)

        self.assertEqual(
            str(context.exception), "The 'target' column contains non-negative entries."
        )


class TestSminaSimulator(TestCase):
    def setUp(self):
        self.incomplete_data = pd.DataFrame({"smiles": [np.nan]})
        self.out = tempfile.TemporaryDirectory()
        self.out_path = pathlib.Path(self.out.name)

    def tearDown(self):
        self.out.cleanup()

    def test_check_simulation_data_no_nan(self):
        with self.assertRaises(ValueError) as context:
            SminaSimulator(self.incomplete_data, self.out_path, "", seed=0)

        self.assertEqual(str(context.exception), "The dataset contains NaN entries.")

    def test_check_simulation_data_has_column(self):
        with self.assertRaises(ValueError) as context:
            SminaSimulator(pd.DataFrame({}), self.out_path, "", seed=0)

        self.assertEqual(
            str(context.exception), "Dataset is missing mandatory columns: smiles."
        )

    def test_output_directory(self):
        not_a_dir = tempfile.NamedTemporaryFile()
        paths = [
            pathlib.Path("/does/not/exist"),
            pathlib.Path(not_a_dir.name),  # Not a directory
        ]

        for path in paths:
            with self.subTest(path=path):
                with self.assertRaises(ValueError) as context:
                    SminaSimulator(pd.DataFrame({"smiles": []}), path, "", seed=0)

                self.assertEqual(
                    str(context.exception), f"{path} is not a valid output directory"
                )

        not_a_dir.close()

    def test_output_directory_temporary_warns(self):
        with self.assertWarns(UserWarning) as context:
            SminaSimulator(pd.DataFrame({"smiles": []}), self.out_path, "", seed=0)
        self.assertEqual(
            str(context.warning),
            f"{self.out_path} appears to be a temporary directory. Output files may be lost.",
        )
