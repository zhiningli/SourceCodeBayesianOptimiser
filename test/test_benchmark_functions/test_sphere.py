import unittest
import numpy as np
from src.utils.benchmark_functions.benchmark_functions import Sphere

class TestSphere(unittest.TestCase):

    def setUp(self):
        """
        Setup a Sphere function instance for testing.
        """
        self.sphere = Sphere(n_dimension=2, noises=0.1, irrelevant_dims=0)

    def test_initialisation(self):
        """
        Test the Sphere function initialization.
        """
        self.assertEqual(self.sphere.n_dimension, 2)
        self.assertEqual(self.sphere.noise_std, 0.1)
        self.assertEqual(self.sphere.irrelevant_dims, 0)
        self.assertEqual(self.sphere.global_minimum, 0)
        np.testing.assert_array_equal(self.sphere.global_minimumX, np.array([0, 0]))
        np.testing.assert_array_equal(self.sphere.search_space, np.array([(-5.12, 5.12), (-5.12, 5.12)]))

    def test_evaluation_1d(self):
        """
        Test the evaluation function with a 1D input (single point).
        """
        X = np.array([1.0, 2.0])
        result = self.sphere.evaluate(X)
        expected = np.sum(X**2)
        self.assertAlmostEqual(result, expected)

    def test_evaluation_2d(self):
        """
        Test the evaluation function with a 2D input (multiple points).
        """
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = self.sphere.evaluate(X)
        expected = np.sum(X**2, axis=1)
        np.testing.assert_almost_equal(result, expected)


    def test_evaluation_nd(self):
        """
        Test the evaluation function with a higher-dimensional input (ndim > 3).
        """
        X = np.random.rand(4, 5, 6)
        result = self.sphere.evaluate(X)
        expected = np.sum(X**2, axis=-1)
        np.testing.assert_almost_equal(result, expected)

    def test_evaluation_dim_mismatch(self):
        """
        Test if the function raises an error for input dimension mismatch.
        """
        X = np.array([1.0])  # Only 1 dimension
        with self.assertRaises(ValueError):
            self.sphere.evaluate(X)

    def test_type_checking(self):
        """
        Test if the function raises an error for incorrect input type.
        """
        X = [1.0, 2.0]  # List instead of numpy array
        with self.assertRaises(TypeError):
            self.sphere.evaluate(X)

    def test_source_code_retrieval(self):
        """
        Test that the source code of the evaluate function is retrieved correctly.
        """
        source_code = self.sphere.get_source_code()
        self.assertIn("np.sum(X**2", source_code)


if __name__ == "__main__":
    unittest.main()
