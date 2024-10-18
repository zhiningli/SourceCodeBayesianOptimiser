import unittest
import numpy as np
from src.surrogate.GP import GP
from src.surrogate.kernels.RBF import RBF


class TestGPRBF(unittest.TestCase):

    def setUp(self):
        self.rbf_kernel = RBF(length_scales=[1.0])

        self.gp = GP(kernel=self.rbf_kernel, noise=1e-2)

        self.X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        self.y_train = np.array([1.0, 2.0, 1.5, 3.0, 2.5])

    def test_initialization(self):
        self.assertEqual(self.gp.noise, 1e-2)
        self.assertEqual(self.gp.kernel, self.rbf_kernel)
        self.assertIsNone(self.gp.X_train)
        self.assertIsNone(self.gp.y_train)

    def test_fit(self):
        self.gp.fit(self.X_train, self.y_train)

        np.testing.assert_array_equal(self.gp.X_train, self.X_train)

        self.assertAlmostEqual(self.gp.y_mean, self.y_train.mean(), places=6)
        self.assertAlmostEqual(self.gp.y_std, max(self.y_train.std(), 1e-8), places=6)

        self.assertIsNotNone(self.gp.L)

    def test_predict(self):
        """Test the GP predict method."""
        self.gp.fit(self.X_train, self.y_train)

        X_test = np.array([[1.5], [3.5]])

        mean, std = self.gp.predict(X_test)

        self.assertEqual(mean.shape, (2,))
        self.assertEqual(std.shape, (2,))

        self.assertTrue(np.all(std > 0))

    def test_cholesky_fallback(self):

        self.gp.noise = 1e10
        self.gp.fit(self.X_train, self.y_train)

        self.assertIsNone(self.gp.L)
        self.assertIsNotNone(self.gp.K_inv)

if __name__ == '__main__':
    unittest.main()
