import unittest.mock
from unittest import TestCase

from bayesian_protein.embedding import (
    FingerprintEmbedder,
    HuggingfaceEmbedder,
    embedding_factory,
)


class TestEmbeddingFactory(TestCase):
    def setUp(self):
        self.batch_size = unittest.mock.Mock()

    def test_embedding_factory_raises(self):
        with self.assertRaises(ValueError) as context:
            embedding_factory("incorrect-arg", self.batch_size)
        self.assertTrue(
            str(context.exception).startswith("Invalid embedding model incorrect-arg.")
        )

    def test_embedding_factory_returns_correct(self):
        cases = [
            ("fingerprint", FingerprintEmbedder),
            ("chemberta-mlm", HuggingfaceEmbedder),
            ("chemberta-mtr", HuggingfaceEmbedder),
        ]

        for model, klass in cases:
            with self.subTest(model=model):
                rval = embedding_factory(model, self.batch_size)
                self.assertIsInstance(rval, klass)

                if model == "chemberta-mlm":
                    self.assertEqual("DeepChem/ChemBERTa-77M-MLM", rval.name)
                elif model == "chemberta-mtr":
                    self.assertEqual("DeepChem/ChemBERTa-77M-MTR", rval.name)
