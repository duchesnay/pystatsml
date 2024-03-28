"""

TOP Module: Supervized models 

"""

import numpy as np
import scipy.linalg

###############################################################################
# Modules and packages
# --------------------


class LinearRegression:
    """Ordinary least squares Linear Regression.

    Application Programming Interface (API) is compliant with scikit-learn:
    fit(X, y), predict(X)

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to False, no intercept will be used in calculations
        (i.e. data is expected to be centered).

    Examples
    --------
    >>> import numpy as np
    >>> from stat_pkg import LinearRegression
    >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> # y = 1 * x_0 + 2 * x_1 + 3
    >>> y = np.dot(X, np.array([1, 2])) + 3
    >>> reg = LinearRegression().fit(X, y)
    >>> reg.coef_
    array([3., 1., 2.0])
    >>> reg.predict(np.array([[3, 5]]))
    array([16.])
    """

    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """Fit linear model.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Design matrix (independent variables, predictors,)
        y : array of shape (n_samples, )
            target (dependent variable, )

        Returns
        -------
        self : object
            Fitted Estimator.
        """
        if self.fit_intercept:
            intercept_ = np.ones((X.shape[0], 1))
            X = np.concatenate((intercept_, X), axis=1)
        self.pinv_ = scipy.linalg.pinv(X)
        self.coef_ = np.dot(self.pinv_, y)

        return self

    def predict(self, X):
        """_summary_

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            New Design matrix (independent variables, predictors,)
        """
        if self.fit_intercept:
            intercept_ = np.ones((X.shape[0], 1))
            X = np.concatenate((intercept_, X), axis=1)
        return np.dot(X, self.coef_)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
