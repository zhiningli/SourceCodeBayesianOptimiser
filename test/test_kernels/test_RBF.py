import unittest
import numpy as np
from src.surrogate.kernels.RBF import RBF


class TestRBFKernel(unittest.TestCase):
    def test_initialization(self):
        length_scales = [1.0, 2.0]
        rbf_kernel = RBF(length_scales)
        np.testing.assert_array_equal(rbf_kernel.length_scales, np.array(length_scales))

    def test_kernel_output(self):
        length_scales = [1.0, 2.0]
        rbf_kernel = RBF(length_scales)
        
        X1 = np.array([[1.0, 2.0]])
        X2 = np.array([[2.0, 3.0]])

        expected_output = np.exp(-0.625)

        result = rbf_kernel(X1, X2)
        self.assertAlmostEqual(result[0, 0], expected_output, places=6)

    def test_kernel_output_multiple_points(self):
        length_scales = [1.0, 1.0]
        rbf_kernel = RBF(length_scales)
        
        X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        X2 = np.array([[5.0, 6.0], [7.0, 8.0]])

        result = rbf_kernel(X1, X2)
        self.assertEqual(result.shape, (2, 2))
    
    def test_string_representation(self):
        length_scales = [1.0, 2.0]
        rbf_kernel = RBF(length_scales)
        self.assertEqual(str(rbf_kernel), "Anisotropic RBF Kernel with length scales [1. 2.]")

if __name__ == '__main__':
    unittest.main()
