import unittest
import numpy as np
from src.utils.benchmark_functions import Beale

class TestBeale(unittest.TestCase):

    def setUp(self):
        """
        Setup a Beale function instance for testing.
        """
        self.beale = Beale(n_dimension=2, noises=0.1, irrelevant_dims=0)

    def test_initialisation(self):
        """
        Test the Beale function initialization.
        """
        self.assertEqual(self.beale.n_dimension, 2)
        self.assertEqual(self.beale.noise_std, 0.1)
        self.assertEqual(self.beale.irrelevant_dims, 0)
        self.assertEqual(self.beale.global_minimum, 0)
        self.assertEqual(self.beale.global_minimumX, [3, 0.5])
        self.assertEqual(self.beale.search_space, [(-4.5, 4.5), (-4.5, 4.5)])

    def test_evaluation_1d(self):
        """
        Test the evaluation function with a 1D input.
        """
        X = np.array([3.0, 0.5])  # The global minimum
        result = self.beale.evaluate(X)
        expected = 0  # Global minimum at (3, 0.5)
        self.assertAlmostEqual(result, expected)

    def test_evaluation_2d(self):
        """
        Test the evaluation function with a 2D input.
        """
        X = np.array([[3.0, 0.5], [1.0, 1.0]])  # Two points
        result = self.beale.evaluate(X)
        x, y = X[:, 0], X[:, 1]
        expected = (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2
        np.testing.assert_almost_equal(result, expected)

    def test_evaluation_dim_mismatch(self):
        """
        Test if the function raises an error for input dimension mismatch.
        """
        X = np.array([1.0])  # Only 1 dimension
        with self.assertRaises(ValueError):
            self.beale.evaluate(X)

    def test_type_checking(self):
        """
        Test if the function raises an error for incorrect input type.
        """
        X = [1.0, 2.0]  # Not a numpy array
        with self.assertRaises(TypeError):
            self.beale.evaluate(X)

    def test_source_code_retrieval(self):
        """
        Test that the source code of the evaluate function is retrieved correctly.
        """
        source_code = self.beale.get_source_code()
        self.assertIn("1.5 - x + x * y", source_code)
        self.assertIn("2.25 - x + x * y**2", source_code)
        self.assertIn("2.625 - x + x * y**3", source_code)


if __name__ == "__main__":
    unittest.main()
