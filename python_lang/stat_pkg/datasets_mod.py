"""

Modules and packages
--------------------

"""
###############################################################################
# Package
# ~~~~~~~
#
# A package is a directory (here, ``datasets_mod``) containing a ``__init__.py`` file.
#
# The ``__init__.py`` can be empty. Or it can be use to define what is seen in
# the package, i.e., (i) import module(s) or functions/classes from the modules in
# the package, and (ii) export them. Example::
#     # 1) import function for modules in the packages
#     from .module import make_regression
#
#     # 2) Make them visible in the package
#     __all__ = ["make_regression"]


###############################################################################
# Module
# ~~~~~~
#
# A module is a python file, example ``module.py`` file:

import numpy as np


def make_regression(n_samples=10, n_features=2, add_intercept=False):
    """Make regression dataset, returns: X, y, coef such that y = X * coef + err
    Parameters
    ----------
    n_samples : int, optional
        number of samples, by default 10
    n_features : int, optional
        number of features, by default 2
    add_intercept : bool, optional
        add intercept, by default False
    Returns
    -------
    X, y, coef: arrays of shapes (n_samples, n_features [+1 if add_intercept]),
    (n_samples, ), and (n_features [+1 if add_intercept], )
        X is the matrix of predictors, y the target (dependant) variable and coef
        is vector of coefficients.
    """
    n_col = n_features + 1 if add_intercept else n_features
    X = np.random.normal(size=n_samples * n_col).reshape((n_samples, n_col))
    coef = np.arange(1, n_col+1)[::-1]
    if add_intercept:
        X[:, 0] = 1
        coef[0] = 1
    err = np.random.normal(size=n_samples)
    y = np.dot(X, coef) + err
    return X, y, coef


