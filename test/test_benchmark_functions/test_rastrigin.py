import unittest
import numpy as np
from src.utils.benchmark_functions import Rastrigin

class TestRastrigin(unittest.TestCase):

    def setUp(self):
        self.rastrigin = Rastrigin(n_dimension=2, noises=0.1, irrelevant_dims=0)

    def test_initialisation(self):
        self.assertEqual(self.rastrigin.n_dimension, 2)
        self.assertEqual(self.rastrigin.noise_std, 0.1)
        self.assertEqual(self.rastrigin.irrelevant_dims, 0)
        self.assertEqual(self.rastrigin.global_minimum, 0)
        np.testing.assert_array_equal(self.rastrigin.global_minimumX, np.array([0, 0]))
        np.testing.assert_array_equal(self.rastrigin.search_space, np.array([(-5.12, 5.12), (-5.12, 5.12)]))

    def test_evaluation_1d(self):
        X = np.array([1.0, 2.0])
        result = self.rastrigin.evaluate(X)
        expected = 3 * 2 + np.sum(X**2 - 3 * np.cos(2 * np.pi * X))  # A = 3
        self.assertAlmostEqual(result, expected)

    def test_evaluation_2d(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = self.rastrigin.evaluate(X)
        expected = 3 * 2 + np.sum(X**2 - 3 * np.cos(2 * np.pi * X), axis=1)
        np.testing.assert_almost_equal(result, expected)

    def test_evaluation_dim_mismatch(self):
        X = np.array([1.0])  
        with self.assertRaises(ValueError):
            self.rastrigin.evaluate(X)

    def test_type_checking(self):
        X = [1.0, 2.0]  # List instead of numpy array
        with self.assertRaises(TypeError):
            self.rastrigin.evaluate(X)

    def test_source_code_retrieval(self):
        source_code = self.rastrigin.get_source_code()
        self.assertIn("A = 3", source_code)
        self.assertIn("return A * X.shape[-1]", source_code)
        self.assertIn("np.sum(X**2 - A * np.cos(2 * np.pi * X)", source_code)



if __name__ == "__main__":
    unittest.main()