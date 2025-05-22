'''
Hands-On: Validation of Supervised Classification Pipelines
===========================================================

'''
################################################################################
# Imports
# -------

# %%

# System
import os
import os.path
import tempfile
import time

# Scientific python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Univariate statistics
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

# Dataset
from sklearn.datasets import make_classification

# Models
from sklearn.decomposition import PCA
import sklearn.linear_model as lm
import sklearn.svm as svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Metrics
import sklearn.metrics as metrics

# Resampling
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline


################################################################################
# Settings
# --------

# %%
# Input/Output and working directory

WD = os.path.join(tempfile.gettempdir(), "ml_supervised_classif")
os.makedirs(WD, exist_ok=True)
INPUT_DIR = os.path.join(WD, "data")
OUTPUT_DIR = os.path.join(WD, "models")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
# Validation scheme here `cross_validation <https://scikit-learn.org/stable/modules/cross_validation.html>`_

n_splits_test = 5
cv_test = StratifiedKFold(n_splits=n_splits_test, shuffle=True, random_state=42)

n_splits_val = 5
cv_val = StratifiedKFold(n_splits=n_splits_val, shuffle=True, random_state=42)

metrics_names = ['accuracy', 'balanced_accuracy', 'roc_auc']


################################################################################
# Dataset
# -------

# %%
X, y = make_classification(n_samples=200, n_features=100,
                           n_informative=10, n_redundant=10,
                           random_state=1)


################################################################################
# Models
# ------

# %%
mlp_param_grid = {"hidden_layer_sizes":
        [(100, ), (50, ), (25, ), (10, ), (5, ),         # 1 hidden layer
        (100, 50, ), (50, 25, ), (25, 10,), (10, 5, ),   # 2 hidden layers
        (100, 50, 25, ), (50, 25, 10, ), (25, 10, 5, )], # 3 hidden layers
        "activation": ["relu"], "solver": ["sgd"], 'alpha': [0.0001]}

models = dict(
    lrl2_cv=make_pipeline(
        preprocessing.StandardScaler(),
        # preprocessing.MinMaxScaler(),
        GridSearchCV(lm.LogisticRegression(),
                     {'C': 10. ** np.arange(-3, 1)},
                     cv=cv_val, n_jobs=n_splits_val)),

    lrenet_cv=make_pipeline(
        preprocessing.StandardScaler(),
        # preprocessing.MinMaxScaler(),
        GridSearchCV(estimator=lm.SGDClassifier(loss='log_loss',
                    penalty='elasticnet'),
                    param_grid={'alpha': 10. ** np.arange(-1, 3),
                                 'l1_ratio': [.1, .5, .9]},
                    cv=cv_val, n_jobs=n_splits_val)),

    svmrbf_cv=make_pipeline(
        # preprocessing.StandardScaler(),
        preprocessing.MinMaxScaler(),
        GridSearchCV(svm.SVC(),
                     # {'kernel': ['poly', 'rbf'], 'C': 10. ** np.arange(-3, 3)},
                     {'kernel': ['rbf'], 'C': 10. ** np.arange(-1, 2)},
                     cv=cv_val, n_jobs=n_splits_val)),

    forest_cv=make_pipeline(
        # preprocessing.StandardScaler(),
        preprocessing.MinMaxScaler(),
        GridSearchCV(RandomForestClassifier(random_state=1),
                     {"n_estimators": [10, 100]},
                     cv=cv_val, n_jobs=n_splits_val)),

    gb_cv=make_pipeline(
        preprocessing.MinMaxScaler(),
        GridSearchCV(estimator=GradientBoostingClassifier(random_state=1),
                     param_grid={"n_estimators": [10, 100]},
                     cv=cv_val, n_jobs=n_splits_val)),

    mlp_cv=make_pipeline(
        # preprocessing.StandardScaler(),
        preprocessing.MinMaxScaler(),
        GridSearchCV(estimator=MLPClassifier(random_state=1, max_iter=200, tol=0.0001),
                     param_grid=mlp_param_grid,
                     cv=cv_val, n_jobs=n_splits_val)))


################################################################################
# Fit/Predict and Compute Test Score (CV)
# ---------------------------------------
#
# Fit/predict models and return scores on folds using: `cross_validate <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html>`_
# 
# Here we set:
#
# - ``return_estimator=True`` to return the estimator fitted on each training set
# - ``return_indices=True`` to return the training and testing indices used to split the dataset into train and test sets for each cv split.

# %%
models_scores = dict()

for name, model in models.items():
    # name, model = "lrl2_cv", models["lrl2_cv"]
    start_time = time.time()
    models_scores_ = cross_validate(estimator=model, X=X, y=y, cv=cv_test,
                                  n_jobs=n_splits_test,
                                  scoring=metrics_names,
                                  return_estimator=True,
                                  return_indices=True)
    print(name, 'Elapsed time: \t%.3f sec' % (time.time() - start_time))
    models_scores[name] = models_scores_


################################################################################
# Average Test Scores (CV) and save it to a file
# ----------------------------------------------

# %%
test_stat = [[name] + [res["test_" + metric].mean() for metric in metrics_names]
             for name, res in models_scores.items()]

test_stat = pd.DataFrame(test_stat, columns=["model"]+metrics_names)
test_stat.to_csv(os.path.join(OUTPUT_DIR, "test_stat.csv"))
print(test_stat)


################################################################################
# Retrieve Individuals Predictions
# --------------------------------

# %%
# **1. Retrieve individuals predictions and save individuals predictions in csv file**

# Iterate over models
predictions = pd.DataFrame()
for name, model in models_scores.items():
    # name, model = "lrl2_cv", models_scores["lrl2_cv"]
    # model_scores = models_scores["lrl2_cv"]

    pred_vals_test = np.full(y.shape, np.nan) # Predicted values before threshold
    pred_vals_train = np.full(y.shape, np.nan) # Predicted values before threshold
    pred_labs_test = np.full(y.shape, np.nan) # Predicted labels
    pred_labs_train = np.full(y.shape, np.nan) # Predicted labels
    true_labs = np.full(y.shape, np.nan) # True labels
    fold_nb = np.full(y.shape, np.nan) # True labels

    # Iterate over folds
    for fold in range(len(model['estimator'])):
        est = model['estimator'][fold]
        test_idx = model['indices']['test'][fold]
        train_idx = model['indices']['train'][fold]
        X_test = X[test_idx]
        X_train = X[train_idx]

        # Predicted labels
        pred_labs_test[test_idx] = est.predict(X_test)
        pred_labs_train[train_idx] = est.predict(X_train)
        fold_nb[test_idx] = fold
        
        # Predicted values before threshold
        try:
            pred_vals_test[test_idx] = est.predict_proba(X_test)[:, 1]
            pred_vals_train[train_idx] = est.predict_proba(X_train)[:, 1]
        except AttributeError:
            pred_vals_test[test_idx] = est.decision_function(X_test)
            pred_vals_train[train_idx] = est.decision_function(X_train)

        true_labs[test_idx] = y[test_idx]

    predictions_ = pd.DataFrame(dict(model=name, fold=fold_nb.astype(int),
                        pred_vals_test=pred_vals_test,
                        pred_labs_test=pred_labs_test.astype(int),
                        true_labs=y))
    assert np.all(true_labs == y)

    predictions = pd.concat([predictions, predictions_])


predictions.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"))

# %%
# **2. Recompute scores from saved predictions**

models_scores_cv = [[mod, fold,
    metrics.balanced_accuracy_score(df["true_labs"], df["pred_labs_test"]),
    metrics.roc_auc_score(df["true_labs"], df["pred_vals_test"])]
 for (mod, fold), df in predictions.groupby(["model", "fold"])]

models_scores_cv = pd.DataFrame(models_scores_cv, columns=["model", "fold", 'balanced_accuracy', 'roc_auc'])

models_scores = models_scores_cv.groupby("model").mean()
models_scores = models_scores.drop("fold", axis=1)
print(models_scores)

