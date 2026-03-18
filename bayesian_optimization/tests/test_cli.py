from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest import TestCase

from bayesian_protein import CommandLineArgs


class TestCommandLineArgs(TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.data = NamedTemporaryFile()
        self.data.write(b"smiles,target")
        self.data.seek(0)

        self.fix = {
            "n_iter": 10,
            "embedding": "fingerprint",
            "out": self.temp_dir.name,
            "data": self.data.name,
            "surrogate": "linear_prior",
            "sampler": "expected-improvement",
            "protein": "",
        }

    def tearDown(self):
        self.temp_dir.cleanup()
        self.data.close()

    def test_out_exists(self):
        f = "/does/not/exist/"
        with self.assertRaises(
            ValueError, msg=f"The output file path '{f}' does not exist."
        ):
            args = self.fix.copy()
            args["out"] = f
            CommandLineArgs(**args)

        try:
            CommandLineArgs(**self.fix)
        except ValueError:
            self.fail()

    def test_out_is_dir(self):
        f = NamedTemporaryFile()
        with self.assertRaises(ValueError) as context:
            args = self.fix
            args["out"] = f.name
            CommandLineArgs(**self.fix)
            f.close()

        self.assertEqual(
            str(context.exception),
            f"The output file path '{f.name} is not a directory.'",
        )

    def test_data_exists(self):
        f = "/does/not/exist"
        with self.assertRaises(ValueError) as context:
            args = self.fix.copy()
            args["data"] = f
            CommandLineArgs(**args)

        self.assertEqual(
            str(context.exception), f"The dataset path '{f}' does not exist."
        )

    def test_cluster_needs_confidence(self):
        with self.assertWarns(Warning) as context:
            args = self.fix.copy()
            args["cluster"] = 3
            CommandLineArgs(**args)

        self.assertEqual(
            str(context.warning.args[0]),
            "--confidence is set to 0 while --cluster is specified",
        )

        try:
            args = self.fix.copy()
            args["cluster"] = 3
            args["confidence"] = 0.1
            CommandLineArgs(**args)
        except ValueError:
            self.fail()

    def test_simulator_and_validate_raises(self):
        with self.assertRaises(ValueError) as context:
            args = self.fix.copy()
            args["simulate"] = "smina"
            args["validate"] = True
            args["protein"] = ""
            CommandLineArgs(**args)

        self.assertEqual(
            str(context.exception),
            "--validate cannot be used with a simulator. "
            "A dataset with measured affinities is required.",
        )
