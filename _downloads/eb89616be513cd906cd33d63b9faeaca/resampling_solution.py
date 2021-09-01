#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 10:40:44 2017

@author: edouard.duchesnay@cea.fr
"""

"""
Exercise

Given the logistic regression presented above and its validation given a 5 folds CV.

    Compute the p-value associated with the prediction accuracy using a permutation test.

    Compute the p-value associated with the prediction accuracy using a parametric test.

"""
import numpy as np
from sklearn import datasets
import sklearn.linear_model as lm
import sklearn.metrics as metrics
from sklearn.model_selection import StratifiedKFold

X, y = datasets.make_classification(n_samples=100, n_features=100,
                         n_informative=10, random_state=42)

model = lm.LogisticRegression(C=1)
nperm = 100
scores_perm= np.zeros((nperm, 3))  # 3 scores acc, recall0, recall1

for perm in range(0, nperm):
    # perm = 0; y == yp
    # first run on non-permuted samples
    yp = y if perm == 0 else np.random.permutation(y)
    # CV loop
    y_test_pred = np.zeros(len(yp))
    cv = StratifiedKFold(5)
    for train, test in cv.split(X, y):
        X_train, X_test, y_train, y_test = X[train, :], X[test, :], yp[train], yp[test]
        model.fit(X_train, y_train)
        y_test_pred[test] = model.predict(X_test)
    scores_perm[perm, 0] = metrics.accuracy_score(yp, y_test_pred)
    scores_perm[perm, [1, 2]] = metrics.recall_score(yp, y_test_pred, average=None)

# Empirical permutation based p-values
pval = np.sum(scores_perm >= scores_perm[0, :], axis=0) / nperm

print("ACC:%.2f(P=%.3f); SPC:%.2f(P=%.3f); SEN:%.2f(P=%.3f)" %\
      (scores_perm[0, 0], pval[0],
       scores_perm[0, 1], pval[1],
       scores_perm[0, 2], pval[2]))

