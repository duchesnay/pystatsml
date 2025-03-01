# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:54:32 2016

@author: ed203246
"""

import numpy as np
X = np.random.randn(4, 2)
print(X)

'''
- For each column find the row indices of the minimiun value.
'''
[np.argmin(X[:, j])
    for j in range(X.shape[1])]

np.argmin(X, axis=0)

'''
- Write a function ``scale(X)`` that return an array whose columns are centered and scaled (by std-dev).
'''

def scale(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

X = np.random.randn(5, 3)
Xs = scale(X)

Xs.mean(axis=0)
Xs.std(axis=0)
