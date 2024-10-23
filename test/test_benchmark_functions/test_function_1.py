import unittest
import numpy as np
from src.utils.benchmark_functions import Function_1

class Test_Function_1(unittest.TestCase):

    def setUp(self):
        self.func = Function_1()


    def test_initialisation(self):
        self.assertEqual(self.func.n_dimension, 7)
        self.assertEqual(self.func.noise_std, 0)
        self.assertEqual(self.func.irrelevant_dims, 0)
        self.assertEqual(self.func.global_minimum, 0.6)
        np.testing.assert_array_equal(self.func.search_space, np.array([(-5.0, 5.0)] * self.func.n_dimension))

    def test_evaluation(self):
        X = np.array([0, 0, 1, 1, 3, 2, -4])
        result = self.func.evaluate(X)
        expected = X[3]**2 + 0.1 + 0.5
        self.assertEqual(result, expected)

    def test_type_checking(self):
        X = [0, 1, 1, 1, 3, 2, -4]
        with self.assertRaises(TypeError):
            self.func.evaluate(X)

    def test_source_code_retrieval(self):
        source_code = self.func.get_source_code()
        self.assertIn("X = X[:7]", source_code)
        self.assertIn("val = X[6]**2 + 0.4 + r9", source_code)
        self.assertIn("return val", source_code)