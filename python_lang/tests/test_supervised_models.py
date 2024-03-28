import unittest
import numpy as np
from stat_pkg import make_regression
from stat_pkg import LinearRegression
import statsmodels.api as sm


class TestLinearRegression(unittest.TestCase):

    def test_fit(self):
        X_inter, y, coef = make_regression(add_intercept=True)
        # Fit with statmodels
        ols_sm = sm.OLS(y, X_inter).fit()

        # fit
        ols = LinearRegression()
        ols.fit(X_inter[:, 1:], y)

        # Test
        # print(ols.coef_, ols_sm.params)
        self.assertTrue(np.allclose(ols.coef_, ols_sm.params))

    def test_predict(self):
        X_inter, y, coef = make_regression(add_intercept=True)

        # Fit with statmodels
        ols_sm = sm.OLS(y, X_inter).fit()
        pred_ols_sm = ols_sm.predict(X_inter)

        # fit
        pred_ols = LinearRegression().fit(
            X_inter[:, 1:], y).predict(X_inter[:, 1:])

        # Test
        # print(ols.coef_, ols_sm.params)
        self.assertTrue(np.allclose(pred_ols_sm, pred_ols))


if __name__ == '__main__':
    unittest.main()
