import unittest
import numpy as np
from scipy.stats import norm
from src.surrogate.GP import GP
from src.acquisition.PI import PI


class TestPI(unittest.TestCase):
    
    def setUp(self):
        self.xi = 0.01
        self.pi = PI(xi=self.xi)
        
        self.gp_mock = GP(kernel=None, noise=1e-2)
        
        self.gp_mock.X_train = np.array([[1.0], [2.0], [3.0]])
        self.gp_mock.y_train = np.array([1.0, 2.0, 1.5])
        
        def mock_predict(X):
            mean = np.array([0.5, 1.0])
            std = np.array([0.1, 0.2]) 
            return mean, std
        
        self.gp_mock.predict = mock_predict

    def test_initialization(self):
        self.assertEqual(self.pi.xi, self.xi)

    def test_compute_pi(self):
        X_test = np.array([[1.5], [2.5]]) 
        
        pi_values = self.pi.compute(X_test, self.gp_mock)
        
        mean, std = self.gp_mock.predict(X_test)
        mean_opt = np.min(self.gp_mock.y_train) 
        
        Z = (mean_opt - mean - self.xi) / std
        expected_pi_values = norm.cdf(Z)

        np.testing.assert_array_almost_equal(pi_values, expected_pi_values, decimal=6)

    def test_pi_near_zero_std(self):
        X_test = np.array([[1.5], [2.5]]) 
        
        def mock_predict_small_std(X):
            mean = np.array([0.5, 1.0])
            std = np.array([1e-12, 1e-12])
            return mean, std
        
        self.gp_mock.predict = mock_predict_small_std

        pi_values = self.pi.compute(X_test, self.gp_mock)
        
        self.assertTrue(np.all(pi_values >= 0))

    def test_edge_case_mean_opt(self):
        X_test = np.array([[1.5], [2.5]])
        
        def mock_predict_high_mean(X):
            mean = np.array([2.0, 3.0])  
            std = np.array([0.5, 0.5])
            return mean, std
        
        self.gp_mock.predict = mock_predict_high_mean

        pi_values = self.pi.compute(X_test, self.gp_mock)

        self.assertTrue(np.all(pi_values >= 0))


if __name__ == '__main__':
    unittest.main()
