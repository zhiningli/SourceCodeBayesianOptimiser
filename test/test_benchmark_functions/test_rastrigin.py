import unittest
import numpy as np
from src.utils.benchmark_functions import Rastrigin

class TestRastrigin(unittest.TestCase):

    def setUp(self):
        self.rastrigin = Rastrigin(n_dimension=2, noises=0.1, irrelevant_dims= 0)
    
    def test_initialisation(self):

        self.assertEqual(self.rastrigin.n_dimension, 2)
        self.assertEqual(self.rastrigin.noise_std, 0.1)
        self.assertEqual(self.rastrigin.irrelevant_dims, 0)
        self.assertEqual(self.rastrigin.global_minimum, 0)
        self.assertEqual(self.rastrigin.global_minimumX, [0, 0])
        self.assertEqual(self.rastrigin.search_space,[(-5.12, 5.12), (-5.12, 5.12)])
    
    def test_evaluation_1d(self):
        X = np.array([1.0, 2.0])
        result = self.rastrigin.evaluate(X)
        expected = 3 * 2 + np.sum(X**2 - 3 * np.cos(2 * np.pi * X))  # A = 3
        self.assertAlmostEqual(result, expected)

    def test_evaluation_2d(self):
        """
        Test the evaluation function with a 2D input.
        """
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = self.rastrigin.evaluate(X)
        expected = 3 * 2 + np.sum(X**2 - 3 * np.cos(2 * np.pi * X), axis=1)
        np.testing.assert_almost_equal(result, expected)

    def test_evaluation_dim_mismatch(self):
        """
        Test if the function raises an error for input dimension mismatch.
        """
        X = np.array([1.0])  
        with self.assertRaises(ValueError):
            self.rastrigin.evaluate(X)

    def test_type_checking(self):
        """
        Test if the function raises an error for incorrect input type.
        """
        X = [1.0, 2.0] 
        with self.assertRaises(TypeError):
            self.rastrigin.evaluate(X)

    def test_source_code_retrieval(self):
        """
        Test that the source code of the evaluate function is retrieved correctly.
        """
        source_code = self.rastrigin.get_source_code()
        self.assertIn("A = 3", source_code)
        self.assertIn("return A * X.shape[-1]", source_code)
        self.assertIn("np.sum(X**2 - A * np.cos(2 * np.pi * X)", source_code)


if __name__ == "__main__":
    unittest.main()