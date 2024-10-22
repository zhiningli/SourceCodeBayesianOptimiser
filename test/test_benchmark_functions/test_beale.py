import unittest
import numpy as np
from src.utils.benchmark_functions import Beale

class TestBeale(unittest.TestCase):

    def setUp(self):
        self.beale = Beale(n_dimension=2, noises=0.1, irrelevant_dims=0)

    def test_initialisation(self):
        self.assertEqual(self.beale.n_dimension, 2)
        self.assertEqual(self.beale.noise_std, 0.1)
        self.assertEqual(self.beale.irrelevant_dims, 0)
        self.assertEqual(self.beale.global_minimum, 0)
        np.testing.assert_array_equal(self.beale.global_minimumX, np.array([3, 0.5]))
        np.testing.assert_array_equal(self.beale.search_space, np.array([(-4.5, 4.5), (-4.5, 4.5)]))

    def test_evaluation_1d(self):
        X = np.array([3.0, 0.5]) 
        result = self.beale.evaluate(X)
        expected = 0 
        self.assertAlmostEqual(result, expected)

    def test_evaluation_2d(self):
        X = np.array([[3.0, 0.5], [1.0, 1.0]])
        x, y = X[:, 0], X[:, 1]
        expected = (1.5 - x + x * y)**2 + (2.25 - x + x * y**2)**2 + (2.625 - x + x * y**3)**2
        result = self.beale.evaluate(X)
        np.testing.assert_almost_equal(result, expected)

    def test_evaluation_dim_mismatch(self):
        X = np.array([1.0]) 
        with self.assertRaises(ValueError):
            self.beale.evaluate(X)

    def test_type_checking(self):
        X = [1.0, 2.0] 
        with self.assertRaises(TypeError):
            self.beale.evaluate(X)

    def test_source_code_retrieval(self):
        source_code = self.beale.get_source_code()
        self.assertIn("1.5 - x + x * y", source_code)
        self.assertIn("2.25 - x + x * y**2", source_code)
        self.assertIn("2.625 - x + x * y**3", source_code)


if __name__ == "__main__":
    unittest.main()
