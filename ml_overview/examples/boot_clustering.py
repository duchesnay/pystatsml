import numpy as np
import scipy
import seaborn as sns
from sklearn import cluster, datasets
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns # nice color
iris = datasets.load_iris()
X = iris.data[:, :2] # use only 'sepal length and sepal width'
y_iris = iris.target
kmr = cluster.KMeans(n_clusters=3, random_state=42).fit(X)
labels_r = kmr.predict(X)
%matplotlib qt

nboot = 500
orig_all = np.arange(X.shape[0])
scores_boot = np.zeros(nboot)
for boot_i in range(nboot):
    # boot_i = 43
    np.random.seed(boot_i)
    boot_idx = np.random.choice(orig_all, size=len(orig_all), replace=True)
    # boot_idx = orig_all
    kmb = cluster.KMeans(n_clusters=3, random_state=42).fit(X[boot_idx, :])
    dist = scipy.spatial.distance.cdist(kmb.cluster_centers_, kmr.cluster_centers_)
    reorder = np.argmin(dist, axis=1)
    #print(reorder)
    # kmb.cluster_centers_ = kmb.cluster_centers_[reorder]
    labels_b = kmb.predict(X)
    labels_b = np.array([reorder[lab] for lab in labels_b])
    scores_boot[boot_i] = np.sum(labels_b == labels_r) / len(labels_b)

sns.distplot(scores_boot)
plt.show()

print(np.min(scores_boot), np.argmin(scores_boot))

pd.Series(scores_boot).describe(percentiles=[.975, .5, .025])
