import unittest
import numpy as np
from stat_pkg import make_regression


class TestDatasets(unittest.TestCase):

    def test_make_regression(self):
        X, y, coefs = make_regression(n_samples=10, n_features=3, add_intercept=True)
        
        # Test        
        self.assertTrue(np.allclose(X.shape, (10, 4)))
        self.assertTrue(np.allclose(y.shape, (10, )))
        self.assertTrue(np.allclose(coefs.shape, (4, )))


if __name__ == '__main__':
    unittest.main()
