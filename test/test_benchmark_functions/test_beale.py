import unittest
import numpy as np
from src.utils.benchmark_functions import Beale

class TestBeale(unittest.TestCase):

    def setUp(self):
        self.beale = Beale(n_dimension=2, noises=0.1)

    
    def test_initialisation(self):

        self.assertEqual(self.beale.n_dimension, 2)
        self.assertEqual(self.beale.noise_std, 0.1)
        self.assertEqual(self.)