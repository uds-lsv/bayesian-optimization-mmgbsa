from unittest import TestCase

from bayesian_protein.surrogates import surrogate_factory


class TestSurrogate(TestCase):
    def test_surrogate_factory_raises(self):
        with self.assertRaises(ValueError) as context:
            surrogate_factory("invalid-surrogate")

        self.assertTrue(
            str(context.exception).startswith(
                f"Invalid surrogate model invalid-surrogate."
            )
        )
