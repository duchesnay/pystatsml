'''
Munivariate statistics exercises
================================
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
np.random.seed(seed=42)  # make the example reproducible

'''
### Dot product and Euclidean norm
'''

a = np.array([2,1])
b = np.array([1,1])

def euclidian(x):
    return np.sqrt(np.dot(x, x))

euclidian(a)

euclidian(a - b)

np.dot(b, a / euclidian(a))

X = np.random.randn(100, 2)
np.dot(X, a / euclidian(a))

'''
### Covariance matrix and Mahalanobis norm
'''

N = 100
mu = np.array([1, 1])
Cov = np.array([[1, .8],
                [.8, 1]])

X = np.random.multivariate_normal(mu, Cov, N)

xbar = np.mean(X, axis=0)
print(xbar)

Xc = (X - xbar)

np.mean(Xc, axis=0)

S = 1 / (N - 1) * np.dot(Xc.T, Xc)
print(S)

#import scipy

Sinv = np.linalg.inv(S)


def mahalanobis(x, xbar, Sinv):
    xc = x - xbar
    return np.sqrt(np.dot(np.dot(xc, Sinv), xc))

dists = pd.DataFrame(
[[mahalanobis(X[i, :], xbar, Sinv),
  euclidian(X[i, :] - xbar)] for i in range(X.shape[0])],
            columns = ['Mahalanobis', 'Euclidean'])

print(dists[:10])

x = X[0, :]

import scipy.spatial
assert(mahalanobis(X[0, :], xbar, Sinv) == scipy.spatial.distance.mahalanobis(xbar, X[0, :], Sinv))
assert(mahalanobis(X[1, :], xbar, Sinv) == scipy.spatial.distance.mahalanobis(xbar, X[1, :], Sinv))




